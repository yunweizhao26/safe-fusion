#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from sle_common import FOLDS, REPORTED, SEED, bootstrap_draws, filled_matrix, interval, load_fills, load_unit
from sle_evaluate_clustering import TREG, cluster, log_normalize

VALUE_ROOT = REPOSITORY / "artifacts" / "paper_evidence" / "review_round3" / "value_v2_ablations" / "value"
CURRENT = "conditional"
MARKERS = ("FOXP3", "IL2RA", "CTLA4")
BUDGETS = ("global", "gene")

def variant_name(value: str) -> str:
    return "Safe Fusion" if value == CURRENT else f"Safe Fusion, {value}"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="set_name", default="main")
    parser.add_argument("--contract-root", type=Path, default=VALUE_ROOT / "contracts")
    parser.add_argument("--selection", type=Path, default=VALUE_ROOT / "selection" / "selection.json")
    parser.add_argument("--values", nargs="+", default=None, help="Values to insert (default: current and chosen).")
    parser.add_argument("--output-dir", type=Path, default=VALUE_ROOT / "sle")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    chosen = json.loads(args.selection.read_text())["chosen"] if args.values is None else None
    values = args.values or list(dict.fromkeys([CURRENT, chosen]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    call_frames, cluster_rows, marker_rows = [], [], []
    for fold in range(FOLDS):
        unit = load_unit(args.set_name, fold, "deploy")
        base = load_fills(unit, "Safe Fusion")
        variants = {}
        for value in values:
            fills = dict(base)
            if value != CURRENT:
                contract = args.contract_root / f"sle_{args.set_name}_{fold}" / value
                metadata = json.loads((contract / "metadata.json").read_text())
                if metadata["cell_ids"] != unit.obs.index.astype(str).tolist() or metadata["gene_ids"] != unit.genes:
                    raise ValueError(f"cell or gene order differs for {contract}")
                fills["value"] = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[base["rows"], base["cols"]], dtype=np.float32)
            variants[variant_name(value)] = fills

        test_rows = np.flatnonzero(unit.test)
        obs = unit.obs.iloc[test_rows]
        author_treg = (obs["cell_type"] == TREG).to_numpy()
        training = ~unit.test
        scaler = StandardScaler().fit(log_normalize(unit.recorded[training]))
        classifier = LogisticRegression(max_iter=2000, C=1.0)
        classifier.fit(scaler.transform(log_normalize(unit.recorded[training])), unit.obs["cell_type"].to_numpy()[training])
        raw = unit.counts[test_rows]
        base_calls = classifier.predict(scaler.transform(log_normalize(raw)))
        base_labels = cluster(raw, 30, 1.0, 0)
        local = {gene: unit.genes.index(gene) for gene in MARKERS}
        runs = [("Unfilled", "none", 0, raw)] + [
            (method, mode, level, filled_matrix(unit, fills, level, mode))
            for method, fills in variants.items() for mode in BUDGETS for level in REPORTED
        ]
        for method, mode, level, matrix in runs:
            calls = base_calls if method == "Unfilled" else classifier.predict(scaler.transform(log_normalize(matrix)))
            before, after = base_calls == TREG, calls == TREG
            frame = pd.DataFrame({"donor": obs["donor"].to_numpy(), "condition": obs["condition"].to_numpy(),
                                  "before": before, "after": after, "into": ~before & after, "out": before & ~after,
                                  "author_treg": author_treg})
            per_donor = frame.groupby(["donor", "condition"])[["before", "after", "into", "out", "author_treg"]].sum().reset_index()
            per_donor.insert(0, "fill_pct", level)
            per_donor.insert(0, "budget", mode)
            per_donor.insert(0, "method", method)
            per_donor.insert(0, "fold", fold)
            call_frames.append(per_donor)
            if method != "Unfilled":
                labels = cluster(matrix, 30, 1.0, 0)
                cluster_rows.append({"fold": fold, "method": method, "budget": mode, "fill_pct": level,
                                     "ari_to_unfilled": adjusted_rand_score(base_labels, labels)})
                changed = (raw == 0) & (matrix != raw)
                for gene, column in local.items():
                    in_other = changed[~author_treg, column]
                    marker_rows.append({"fold": fold, "method": method, "budget": mode, "fill_pct": level, "gene": gene,
                                        "filled_non_treg_cells": int(in_other.sum()),
                                        "inserted_sum": float(matrix[~author_treg, column][in_other].sum()),
                                        "filled_moved_into_treg": int((changed[:, column] & ~before & after).sum())})
            print(f"fold {fold} {method} {mode} {level}", flush=True)

    counts = pd.concat(call_frames, ignore_index=True)
    counts.to_csv(args.output_dir / "treg_calls_by_donor.csv", index=False)
    clusters = pd.DataFrame(cluster_rows)
    clusters.to_csv(args.output_dir / "clustering_by_fold.csv", index=False)
    markers = pd.DataFrame(marker_rows).groupby(["method", "budget", "fill_pct", "gene"], sort=False).sum(numeric_only=True).reset_index()
    markers["mean_inserted_value"] = markers["inserted_sum"] / markers["filled_non_treg_cells"].where(markers["filled_non_treg_cells"] > 0)
    markers.drop(columns=["fold", "inserted_sum"]).to_csv(args.output_dir / "marker_fills.csv", index=False)

    donors = counts.loc[counts["method"] == "Unfilled"].drop_duplicates("donor").set_index("donor")["condition"].sort_index()
    weights = np.vstack([np.ones(len(donors)), bootstrap_draws(donors.index.to_numpy(), donors.to_numpy(), args.draws, SEED)])
    base = counts.loc[counts["method"] == "Unfilled"].set_index("donor").reindex(donors.index)
    changes: dict[tuple, np.ndarray] = {}
    summary_rows = []
    for (method, mode, level), frame in counts.groupby(["method", "budget", "fill_pct"], sort=False):
        frame = frame.set_index("donor").reindex(donors.index)
        before = weights @ base["before"].to_numpy(float)
        after = weights @ frame["after"].to_numpy(float)
        change = (after - before) / before
        changes[(method, mode, level)] = change
        summary_rows.append({"method": method, "budget": mode, "fill_pct": level,
                             "treg_calls_before": int(before[0]), "treg_calls_after": int(after[0]),
                             "relative_change": float(change[0]), "relative_change_ci": interval(change[1:]),
                             "into_treg": int(frame["into"].sum()), "out_of_treg": int(frame["out"].sum())})
    for value in values:
        if value == CURRENT:
            continue
        for mode in BUDGETS:
            for level in REPORTED:
                difference = changes[(variant_name(value), mode, level)] - changes[(variant_name(CURRENT), mode, level)]
                summary_rows.append({"method": f"{variant_name(value)} minus Safe Fusion", "budget": mode, "fill_pct": level,
                                     "relative_change": float(difference[0]), "relative_change_ci": interval(difference[1:])})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "treg_calls_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(clusters.groupby(["method", "budget", "fill_pct"], sort=False)["ari_to_unfilled"].mean().to_string())
    print(markers.to_string(index=False))

if __name__ == "__main__":
    main()
