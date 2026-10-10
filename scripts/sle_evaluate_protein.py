#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sle_common import CASE, FOLDS, METHODS, NOT_FILLED, REPORTED, SEED, bootstrap_draws, interval, load_fills, load_unit, pair_auroc, weighted_auroc

TAG = "CD307c/FcRL3"
GENE = "FCRL3"
QUANTILES = np.linspace(0.30, 0.70, 41)

def weighted_spearman(x: np.ndarray, y: np.ndarray, item_weights: np.ndarray) -> np.ndarray:

    rx, ry = stats.rankdata(x), stats.rankdata(y)
    total = item_weights.sum(axis=1, keepdims=True)
    mx = item_weights @ rx / total[:, 0]
    my = item_weights @ ry / total[:, 0]
    cov = item_weights @ (rx * ry) / total[:, 0] - mx * my
    vx = item_weights @ (rx * rx) / total[:, 0] - mx**2
    vy = item_weights @ (ry * ry) / total[:, 0] - my**2
    return cov / np.sqrt(vx * vy)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", default="cite")
    parser.add_argument("--adt", type=Path, default=CASE / "cite" / "adt.parquet")
    parser.add_argument("--output-dir", type=Path, default=CASE / "results" / "q7_protein")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    adt = pd.read_parquet(args.adt, columns=["cell_id", f"clr|{TAG}", f"count|{TAG}"]).set_index("cell_id")

    frames, detected_frames = [], []
    for fold in range(FOLDS):
        unit = load_unit(args.set_name, fold, "deploy")
        g = unit.genes.index(GENE)
        fills = {method: load_fills(unit, method) for method in METHODS}
        rows, cols = fills["Safe Fusion"]["rows"], fills["Safe Fusion"]["cols"]
        in_gene = np.flatnonzero(cols == g)
        cell_rows = rows[in_gene]
        cell_ids = unit.obs.index.to_numpy()[cell_rows]
        frame = pd.DataFrame({"fold": fold, "donor": unit.obs["donor"].to_numpy()[cell_rows],
                              "lineage": unit.obs["lineage"].to_numpy()[cell_rows],
                              "protein": adt.loc[cell_ids, f"clr|{TAG}"].to_numpy(),
                              "protein_count": adt.loc[cell_ids, f"count|{TAG}"].to_numpy(),
                              "Log library size": np.log1p(unit.counts[cell_rows].sum(axis=1))})
        for method, data in fills.items():
            frame[method] = data["priority"][in_gene]
            for mode in ("global", "gene"):
                frame[f"first_{mode}|{method}"] = np.where(data["value"][in_gene] > 0, data[f"first_{mode}"][in_gene], NOT_FILLED)
        frames.append(frame)
        test_rows = np.flatnonzero(unit.test & (unit.counts[:, g] > 0))
        library = unit.counts[test_rows].sum(axis=1)
        detected_frames.append(pd.DataFrame({
            "donor": unit.obs["donor"].to_numpy()[test_rows], "lineage": unit.obs["lineage"].to_numpy()[test_rows],
            "rna_cp10k": 1e4 * unit.counts[test_rows, g] / library,
            "protein": adt.loc[unit.obs.index.to_numpy()[test_rows], f"clr|{TAG}"].to_numpy()}))
    zeros = pd.concat(frames, ignore_index=True)
    detected = pd.concat(detected_frames, ignore_index=True)
    donors = np.asarray(sorted(zeros["donor"].unique()))
    weights = np.vstack([np.ones(len(donors)), bootstrap_draws(donors, None, args.draws, SEED)])
    scorers = [*METHODS, "Log library size"]

    rows_out, positive_rows = [], []
    scopes = {"All cells": np.ones(len(zeros), bool), **{name: (zeros["lineage"] == name).to_numpy() for name in sorted(zeros["lineage"].unique())}}
    for scope, members in scopes.items():
        frame = zeros.loc[members].reset_index(drop=True)
        donor_index = pd.Index(donors).get_indexer(frame["donor"])
        item_weights = weights[:, donor_index]
        protein = frame["protein"].to_numpy(float)
        rho = {name: weighted_spearman(frame[name].to_numpy(float), protein, item_weights) for name in scorers}
        for name in scorers:
            score = frame[name].to_numpy(float)
            aurocs = []
            for quantile in QUANTILES:
                high = protein > np.quantile(protein, quantile)
                pos = [score[high & (donor_index == d)] for d in range(len(donors))]
                neg = [score[~high & (donor_index == d)] for d in range(len(donors))]
                n_pos = np.asarray([len(v) for v in pos], float)
                n_neg = np.asarray([len(v) for v in neg], float)
                aurocs.append(weighted_auroc(pair_auroc(pos, neg), n_pos, n_neg, weights, weights))
            auroc = np.nanmean(aurocs, axis=0)
            row = {"scope": scope, "method": name, "rna_zero_cells": int(len(frame)), "donors": int(frame["donor"].nunique()),
                   "spearman": float(rho[name][0]), "spearman_ci": interval(rho[name][1:]),
                   "mean_auroc_q30_q70": float(auroc[0]), "mean_auroc_q30_q70_ci": interval(auroc[1:])}
            if name != "Safe Fusion":
                difference = rho["Safe Fusion"] - rho[name]
                row["safe_fusion_minus_method_spearman"] = float(difference[0])
                row["safe_fusion_minus_method_spearman_ci"] = interval(difference[1:])
            if name in METHODS:
                for mode in ("global", "gene"):
                    first = frame[f"first_{mode}|{name}"].to_numpy()
                    for level in REPORTED:
                        filled = first <= level
                        selected = item_weights @ (filled * protein) / np.maximum(item_weights @ filled, 1e-12)
                        background = item_weights @ protein / item_weights.sum(axis=1)
                        enrichment = np.where(item_weights @ filled > 0, selected - background, np.nan)
                        row[f"{mode}_filled_{level}pct"] = int(filled.sum())
                        row[f"{mode}_protein_enrichment_{level}pct"] = float(enrichment[0])
                        row[f"{mode}_protein_enrichment_{level}pct_ci"] = interval(enrichment[1:])
            rows_out.append(row)
        block = detected.loc[detected["lineage"] == scope] if scope != "All cells" else detected
        if len(block) >= 3:
            d_index = pd.Index(donors).get_indexer(block["donor"])
            control = weighted_spearman(block["rna_cp10k"].to_numpy(float), block["protein"].to_numpy(float), weights[:, d_index])
            positive_rows.append({"scope": scope, "rna_detected_cells": int(len(block)),
                                  "spearman_rna_vs_protein": float(control[0]), "ci": interval(control[1:]),
                                  "mean_protein_rna_detected": float(block["protein"].mean()),
                                  "mean_protein_rna_zero": float(zeros.loc[members, "protein"].mean())})
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows_out)
    table.to_csv(output / "fcrl3_protein_agreement.csv", index=False)
    pd.DataFrame(positive_rows).to_csv(output / "fcrl3_positive_control.csv", index=False)
    summary = {"tag": TAG, "gene": GENE, "donors": donors.tolist(), "rna_zero_cells": int(len(zeros)),
               "rna_detected_cells": int(len(detected))}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table[["scope", "method", "rna_zero_cells", "spearman", "spearman_ci", "mean_auroc_q30_q70"]].to_string(index=False))
    print(pd.DataFrame(positive_rows).to_string(index=False))

if __name__ == "__main__":
    main()
