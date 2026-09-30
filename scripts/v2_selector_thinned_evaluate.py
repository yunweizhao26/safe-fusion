#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_thinning_transfer import FRACTIONS, STRATA, datasets, dense, unit_table
from selector_attribution import exact_topk

CURRENT = "Safe Fusion, mask-trained"

def new_selector_counts(table: dict, name: str, scores: np.ndarray, unit: dict, truth_counts: np.ndarray,
                        thinned: np.ndarray, cell_ids: np.ndarray, bio_units: np.ndarray) -> None:

    split = pd.read_parquet(unit["splits"]).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = np.flatnonzero(split == "test")
    local_rows, cols = np.where(thinned[test] == 0)
    rows = test[local_rows]
    if len(scores) != len(rows):
        raise ValueError(f"{unit['key']}: {len(scores)} scores for {len(rows)} candidates")
    original = truth_counts[rows, cols]
    labels = original > 0
    n = len(labels)
    unique = np.unique(bio_units[rows])
    index = np.searchsorted(unique, bio_units[rows])
    if [f"{unit['key']}:{u}" for u in unique] != table["units"]:
        raise ValueError(f"{unit['key']}: unit order differs")

    def per_unit(weights: np.ndarray) -> np.ndarray:
        return np.bincount(index, weights=weights, minlength=len(unique))

    selections = {b: exact_topk(scores, max(1, int(round(b * n)))) for b in FRACTIONS}
    count_one = labels & (original == 1)
    table["counts"][name] = {b: (per_unit(s), per_unit(s & labels)) for b, s in selections.items()}
    table["count_one_hits"][name] = per_unit(selections[0.05] & count_one)
    table["strata"][name] = {label: (int((selections[0.05] & labels & (original >= lo) & (original <= hi)).sum()),
                                     int((labels & (original >= lo) & (original <= hi)).sum()))
                             for lo, hi, label in STRATA}
    table["average_precision"][name] = float(average_precision_score(labels, scores))

def analyze(tables: list[dict], references: list[str], draws: int, seed: int) -> tuple[list[dict], list[dict]]:
    units = sum((t["units"] for t in tables), [])
    positives = np.concatenate([t["positives"] for t in tables])
    names = list(tables[0]["counts"])
    counts = {name: {b: tuple(np.concatenate([t["counts"][name][b][i] for t in tables]) for i in (0, 1))
                     for b in FRACTIONS} for name in names}
    rng = np.random.default_rng(seed)
    index = np.vstack([np.arange(len(units)), rng.integers(0, len(units), size=(draws, len(units)))])

    def mean_f1(name: str) -> np.ndarray:
        return np.mean([2.0 * counts[name][b][1][index].sum(-1) / (counts[name][b][0][index].sum(-1) + positives[index].sum(-1))
                        for b in FRACTIONS], axis=0)

    f1 = {name: mean_f1(name) for name in names}
    rows = []
    for name in names:
        row = {"method": name, "mean_f1_1_10": 100 * float(f1[name][0]),
               "mean_f1_low": 100 * float(np.quantile(f1[name][1:], 0.025)),
               "mean_f1_high": 100 * float(np.quantile(f1[name][1:], 0.975)),
               "average_precision": 100 * float(np.mean([t["average_precision"][name] for t in tables]))}
        for _, _, label in STRATA:
            hit = sum(t["strata"][name][label][0] for t in tables)
            total = sum(t["strata"][name][label][1] for t in tables)
            row[f"recall_at_5_count_{label}"] = 100 * hit / max(1, total)
        rows.append(row)
    comparisons = []
    for reference in references:
        for other in names:
            if other == reference:
                continue
            difference = f1[reference] - f1[other]
            comparisons.append({"reference": reference, "comparator": other, "metric": "mean_f1_1_10",
                                "difference": 100 * float(difference[0]),
                                "low": 100 * float(np.quantile(difference[1:], 0.025)),
                                "high": 100 * float(np.quantile(difference[1:], 0.975))})
    return rows, comparisons

def thinned_datasets(root: Path, thinning: Path, colon_root: Path) -> dict[str, list[dict]]:

    result = {}
    for key, dataset in datasets(root, thinning).items():
        if key.startswith("colon"):
            continue
        result[key] = [{**unit, "data": dataset["data"], "truth": dataset["truth"], "unit_column": dataset["unit_column"],
                        "comparators": dataset["comparators"]} for unit in dataset["units"]]
    crossfit = colon_root / "thinning"
    for level in ("colon_thinning_050", "colon_thinning_025"):
        result[level] = [{
            "key": f"{level}_{fold}", "splits": colon_root / f"fold_{fold}" / "splits.parquet",
            "thin_root": crossfit / "thinning_trained" / f"{level}_{fold}",
            "thin_stacked": crossfit / "thinning_trained/stacked" / f"{level}_{fold}",
            "mask_root": crossfit / "mask_trained" / f"{level}_{fold}",
            "data": crossfit / "data" / f"{level}_{fold}", "truth": colon_root / f"fold_{fold}" / "prepared.h5ad",
            "unit_column": "donor", "comparators": crossfit / "thinning_trained/comparators" / f"{level}_{fold}",
        } for fold in range(3)]
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round2/thinning_transfer")
    parser.add_argument("--thinning-root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/thinning")
    parser.add_argument("--colon-root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round3/colon_crossfit")
    parser.add_argument("--v2-scores", nargs=2, action="append", metavar=("NAME", "ROOT"), required=True,
                        help="Method name and root with <thinned unit>/test_scores.npy of a new selector.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    new = {name: Path(root) for name, root in args.v2_scores}
    references = [*new, CURRENT]
    frames, differences = [], []
    for key, units in thinned_datasets(args.root, args.thinning_root, args.colon_root).items():
        tables = []
        for unit in units:
            data = ad.read_h5ad(unit["data"] / "corrupted.h5ad")
            truth = ad.read_h5ad(unit["truth"])
            if not np.array_equal(data.obs_names, truth.obs_names) or not np.array_equal(data.var_names, truth.var_names):
                raise ValueError(f"{unit['key']}: thinned and truth orders differ")
            thinned = dense(data.layers["corrupted_counts"]).astype(np.float32)
            truth_counts = dense(truth.layers["counts"]).astype(np.float32)
            cell_ids = data.obs_names.astype(str).to_numpy()
            bio_units = truth.obs[unit["unit_column"]].astype(str).to_numpy()
            table = unit_table(unit, unit, truth_counts, thinned, cell_ids, bio_units)
            for name, root in new.items():
                scores = np.load(root / unit["key"] / "test_scores.npy")
                new_selector_counts(table, name, scores, unit, truth_counts, thinned, cell_ids, bio_units)
            tables.append(table)
        rows, comparisons = analyze(tables, references, args.draws, args.seed)
        frames.append(pd.DataFrame(rows).assign(dataset=key))
        differences.append(pd.DataFrame(comparisons).assign(dataset=key))
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(key)
            print(frames[-1].round(2).to_string(index=False))
            print(differences[-1].round(2).to_string(index=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "summary.csv", index=False, float_format="%.4f")
    pd.concat(differences, ignore_index=True).to_csv(args.output_dir / "paired_differences.csv", index=False,
                                                     float_format="%.4f")
    (args.output_dir / "design.json").write_text(json.dumps({
        "new_selectors": {name: str(root) for name, root in new.items()}, "current": CURRENT,
        "colon_root": str(args.colon_root),
        "draws": args.draws, "seed": args.seed, "fractions": FRACTIONS}, indent=2) + "\n")

if __name__ == "__main__":
    main()
