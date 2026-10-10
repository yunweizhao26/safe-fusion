#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score

REPOSITORY = Path(__file__).resolve().parents[1]
COL = REPOSITORY / "artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial"
PAN = REPOSITORY / "artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets"
CF = REPOSITORY / "artifacts/paper_evidence/pancreas_crossfit"
FRACTIONS = [i / 100 for i in range(1, 11)]
STRATA = [(1, 1, "1"), (2, 5, "2-5"), (6, np.inf, ">5")]
TEACHERS = [
    ("Stacked value", "safe_fusion"),
    ("SVD", "svd_impute"),
    ("Weighted kNN", "graph_smooth"),
    ("Gene median", "gene_median"),
    ("MAGIC teacher (inductive)", "magic_inductive"),
    ("scVI teacher (inductive)", "scvi_inductive"),
]
DATASETS = {
    "colon_thinning_050": [("colon_thinning_050", COL / "splits.parquet", COL / "preprocessed.h5ad")],
    "colon_thinning_025": [("colon_thinning_025", COL / "splits.parquet", COL / "preprocessed.h5ad")],
    "pancreas_thinning_050": [
        (f"pancreas_thinning_050_{k}", CF / f"fold_{k}/splits.parquet", PAN / "preprocessed.h5ad") for k in range(3)
    ],
}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def load(path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r")[rows, cols], np.float32)


def unit_tables(root: Path, dataset: str, unit: str, splits_path: Path, truth_counts: np.ndarray,
                thinned: np.ndarray, cell_ids: np.ndarray, donors: np.ndarray) -> dict:
    split = pd.read_parquet(splits_path).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = np.flatnonzero(split == "test")
    training = np.isin(split, ["development", "validation"])
    local_rows, cols = np.where(thinned[test] == 0)
    rows = test[local_rows]
    original = truth_counts[rows, cols]
    labels = original > 0
    n = len(labels)
    library = thinned.sum(axis=1)
    gene_rate = thinned[training].sum(axis=0) / thinned[training].sum()
    methods = root / unit
    comparators = root / "comparators" / dataset
    scores = {name: load(methods / sub / "mean.npy", rows, cols) for name, sub in TEACHERS}
    scores["MAGIC (all cells)"] = np.expm1(load(comparators / "magic/mean.npy", rows, cols)) * library[rows]
    scores["scVI (all cells)"] = load(comparators / "scvi/mean.npy", rows, cols) * library[rows]
    scores["Detection rate x depth"] = (1.0 - np.exp(-library[rows] * gene_rate[cols])).astype(np.float32)

    selections: dict[str, dict[float, np.ndarray]] = {}
    average_precision = {}
    for name, score in scores.items():
        order = np.lexsort((np.random.default_rng(1729).random(n), -score))
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n)
        selections[name] = {b: rank < max(1, int(round(b * n))) for b in FRACTIONS}
        average_precision[name] = float(average_precision_score(labels, score))
    selections["Safe Fusion"] = {}
    for b in FRACTIONS:
        contract = methods / "selector" / f"safe_fusion_calibrated_mlp_topk_{b:g}".replace(".", "p")
        selected = load(contract / "mean.npy", rows, cols) > 0
        if abs(int(selected.sum()) - max(1, int(round(b * n)))) > 1:
            raise ValueError(f"{unit}: Safe Fusion selects {int(selected.sum())} zeros at b={b}")
        selections["Safe Fusion"][b] = selected
    report = json.loads((methods / "selector/calibration_report.json").read_text())
    average_precision["Safe Fusion"] = float(report["test"]["pr_auc"])

    unit_donors = donors[rows]
    unique = np.unique(unit_donors)
    index = np.searchsorted(unique, unit_donors)
    counts_by_donor = {
        name: {b: (np.bincount(index, weights=s, minlength=len(unique)),
                   np.bincount(index, weights=s & labels, minlength=len(unique))) for b, s in by_b.items()}
        for name, by_b in selections.items()
    }
    strata = {
        name: {label: (int((by_b[0.05] & labels & (original >= lo) & (original <= hi)).sum()),
                       int((labels & (original >= lo) & (original <= hi)).sum()))
               for lo, hi, label in STRATA}
        for name, by_b in selections.items()
    }
    return {"donors": list(unique), "positives": np.bincount(index, weights=labels, minlength=len(unique)),
            "counts": counts_by_donor, "strata": strata, "n": n, "original": original[labels],
            "average_precision": average_precision}


def f1(selected: np.ndarray, hits: np.ndarray, positives: np.ndarray) -> float:
    return 2.0 * hits.sum() / (selected.sum() + positives.sum())


def analyze(units: list[dict], bootstrap: int, seed: int) -> dict:
    donors = sum((u["donors"] for u in units), [])
    positives = np.concatenate([u["positives"] for u in units])
    names = list(units[0]["counts"])
    counts = {name: {b: tuple(np.concatenate([u["counts"][name][b][i] for u in units]) for i in (0, 1))
                     for b in FRACTIONS} for name in names}
    rng = np.random.default_rng(seed)
    draws = [rng.integers(0, len(donors), len(donors)) for _ in range(bootstrap)]
    everyone = np.arange(len(donors))

    def mean_difference(name: str, index: np.ndarray, fractions) -> float:
        return float(np.mean([
            f1(counts["Safe Fusion"][b][0][index], counts["Safe Fusion"][b][1][index], positives[index])
            - f1(counts[name][b][0][index], counts[name][b][1][index], positives[index])
            for b in fractions
        ]))

    rows = []
    for name in names:
        curve = [100 * f1(*counts[name][b], positives) for b in FRACTIONS]
        row = {"method": name,
               "average_precision": 100 * float(np.mean([u["average_precision"][name] for u in units])),
               "f1_1": curve[0], "f1_5": curve[4], "f1_10": curve[9], "mean_f1_1_10": float(np.mean(curve))}
        for _, _, label in STRATA:
            hit = sum(u["strata"][name][label][0] for u in units)
            total = sum(u["strata"][name][label][1] for u in units)
            row[f"recall_at_5_count_{label}"] = 100 * hit / max(1, total)
        if name != "Safe Fusion":
            for label, fractions in [("mean_1_10", FRACTIONS), ("at_1", [0.01]), ("at_5", [0.05]), ("at_10", [0.10])]:
                point = mean_difference(name, everyone, fractions)
                sample = [mean_difference(name, draw, fractions) for draw in draws]
                row[f"sf_minus_{label}"] = 100 * point
                row[f"sf_minus_{label}_low"] = 100 * float(np.quantile(sample, 0.025))
                row[f"sf_minus_{label}_high"] = 100 * float(np.quantile(sample, 0.975))
        rows.append(row)
    original = np.concatenate([u["original"] for u in units])
    n = sum(u["n"] for u in units)
    composition = {
        "test_candidates": int(n), "positives": int(len(original)), "prevalence": float(len(original) / n),
        "positive_count_1_share": float(np.mean(original == 1)),
        "positive_count_2_5_share": float(np.mean((original >= 2) & (original <= 5))),
        "positive_count_gt5_share": float(np.mean(original > 5)),
        "test_donors": len(donors),
    }
    return {"composition": composition, "rows": sorted(rows, key=lambda r: -r["mean_f1_1_10"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="artifacts/paper_evidence/thinning")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root = Path(args.root)
    results, frames = {}, []
    for dataset, units in DATASETS.items():
        data = ad.read_h5ad(root / "data" / dataset / "corrupted.h5ad")
        truth = ad.read_h5ad(units[0][2])
        if not np.array_equal(data.obs_names, truth.obs_names) or not np.array_equal(data.var_names, truth.var_names):
            raise ValueError(f"{dataset}: thinned and truth orders differ")
        thinned = dense(data.layers["corrupted_counts"]).astype(np.float32)
        truth_counts = dense(truth.layers["counts"]).astype(np.float32)
        cell_ids = data.obs_names.astype(str).to_numpy()
        donors = truth.obs["donor"].astype(str).to_numpy()
        tables = [unit_tables(root, dataset, unit, splits, truth_counts, thinned, cell_ids, donors) for unit, splits, _ in units]
        results[dataset] = analyze(tables, args.bootstrap, args.seed)
        frame = pd.DataFrame(results[dataset]["rows"])
        frame.insert(0, "dataset", dataset)
        frames.append(frame)
        print(f"== {dataset}: {results[dataset]['composition']}")
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(frame.drop(columns="dataset").round(2).to_string(index=False))
    output = root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output / "matched_fraction_summary.csv", index=False)
    (output / "matched_fraction_summary.json").write_text(json.dumps(results, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
