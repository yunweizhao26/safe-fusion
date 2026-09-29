#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import average_precision_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from selector_attribution import exact_topk

COL = REPOSITORY / "artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial"
PAN = REPOSITORY / "artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets"
CF = REPOSITORY / "artifacts/paper_evidence/pancreas_crossfit"
NORMAN = REPOSITORY / "artifacts/paper_evidence/review_round2/leakage_free/norman_crispra"
FRACTIONS = [i / 100 for i in range(1, 11)]
STRATA = [(1, 1, "1"), (2, 5, "2-5"), (6, np.inf, ">5")]
DESIGNS = ("thinning-trained", "mask-trained")
SELECTORS = {"Safe Fusion": None, "Selector on scVI": "scvi", "Selector on MAGIC": "magic"}
UNTRAINED = ("scVI", "MAGIC", "SVD", "Expected count")


def datasets(root: Path, thinning: Path) -> dict[str, dict]:
    def spec(data: Path, truth: Path, unit_column: str, comparators: Path, units: list[tuple[str, Path, Path]]) -> dict:
        return {"data": data, "truth": truth, "unit_column": unit_column, "comparators": comparators,
                "units": [{"key": key, "splits": splits, "thin_root": thin_root,
                           "thin_stacked": root / "thinning_trained/stacked" / key,
                           "mask_root": root / "mask_trained" / key} for key, splits, thin_root in units]}
    colon = {
        key: spec(thinning / "data" / key, COL / "preprocessed.h5ad", "donor", thinning / "comparators" / key,
                  [(key, COL / "splits.parquet", thinning / key)])
        for key in ("colon_thinning_050", "colon_thinning_025")
    }
    pancreas = spec(
        thinning / "data/pancreas_thinning_050", PAN / "preprocessed.h5ad", "donor",
        thinning / "comparators/pancreas_thinning_050",
        [(f"pancreas_thinning_050_{k}", CF / f"fold_{k}/splits.parquet", thinning / f"pancreas_thinning_050_{k}") for k in range(3)],
    )
    norman = spec(
        root / "data/norman_thinning_050", NORMAN / "prepared.h5ad", "target",
        root / "thinning_trained/comparators/norman_thinning_050",
        [("norman_thinning_050", NORMAN / "splits.parquet", root / "thinning_trained/norman_thinning_050")],
    )
    return {**colon, "pancreas_thinning_050": pancreas, "norman_thinning_050": norman}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def load(path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r")[rows, cols], np.float32)


def selector_selections(selector: Path, rows: np.ndarray, cols: np.ndarray, n: int) -> dict[float, np.ndarray]:
    selections = {}
    for b in FRACTIONS:
        contract = selector / f"safe_fusion_calibrated_mlp_topk_{b:g}".replace(".", "p")
        selected = load(contract / "mean.npy", rows, cols) > 0
        if abs(int(selected.sum()) - max(1, int(round(b * n)))) > 1:
            raise ValueError(f"{selector}: {int(selected.sum())} zeros selected at b={b}")
        selections[b] = selected
    return selections


def stacked_selections(directory: Path, rows: np.ndarray, cols: np.ndarray) -> tuple[dict[float, np.ndarray], float]:
    saved = np.load(directory / "test_scores.npz")
    if not (np.array_equal(saved["rows"], rows) and np.array_equal(saved["cols"], cols)):
        raise ValueError(f"{directory}: candidate zeros differ from the thinned test zeros")
    score = saved["score"]
    n = len(score)
    report = json.loads((directory / "report.json").read_text())
    return {b: exact_topk(score, max(1, int(round(b * n)))) for b in FRACTIONS}, float(report["test"]["pr_auc"])


def unit_table(dataset: dict, unit: dict, truth_counts: np.ndarray, thinned: np.ndarray,
               cell_ids: np.ndarray, bio_units: np.ndarray) -> dict:
    split = pd.read_parquet(unit["splits"]).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = np.flatnonzero(split == "test")
    training = np.isin(split, ["development", "validation"])
    local_rows, cols = np.where(thinned[test] == 0)
    rows = test[local_rows]
    original = truth_counts[rows, cols]
    labels = original > 0
    n = len(labels)
    library = thinned.sum(axis=1)
    gene_rate = thinned[training].sum(axis=0) / thinned[training].sum()

    scores = {
        "scVI": load(dataset["comparators"] / "scvi/mean.npy", rows, cols) * library[rows],
        "MAGIC": np.expm1(load(dataset["comparators"] / "magic/mean.npy", rows, cols)) * library[rows],
        "SVD": load(unit["thin_root"] / "svd_impute/mean.npy", rows, cols),
        "Expected count": (1.0 - np.exp(-library[rows] * gene_rate[cols])).astype(np.float32),
    }
    selections: dict[str, dict[float, np.ndarray]] = {}
    average_precision: dict[str, float] = {}
    for name, score in scores.items():
        order = np.lexsort((np.random.default_rng(1729).random(n), -score))
        rank = np.empty(n, np.int64)
        rank[order] = np.arange(n)
        selections[name] = {b: rank < max(1, int(round(b * n))) for b in FRACTIONS}
        average_precision[name] = float(average_precision_score(labels, score))
    for design in DESIGNS:
        root = unit["thin_root"] if design == "thinning-trained" else unit["mask_root"]
        for selector, contract in SELECTORS.items():
            name = f"{selector}, {design}"
            if contract is None:
                selections[name] = selector_selections(root / "selector", rows, cols, n)
                report = json.loads((root / "selector/calibration_report.json").read_text())
                average_precision[name] = float(report["test"]["pr_auc"])
            else:
                stacked = unit["thin_stacked"] if design == "thinning-trained" else root / "stacked"
                selections[name], average_precision[name] = stacked_selections(stacked / contract, rows, cols)

    unit_labels = bio_units[rows]
    unique = np.unique(unit_labels)
    index = np.searchsorted(unique, unit_labels)

    def per_unit(weights: np.ndarray) -> np.ndarray:
        return np.bincount(index, weights=weights, minlength=len(unique))

    count_one = labels & (original == 1)
    return {
        "units": [f"{unit['key']}:{u}" for u in unique],
        "positives": per_unit(labels),
        "count_one": per_unit(count_one),
        "counts": {name: {b: (per_unit(s), per_unit(s & labels)) for b, s in by_b.items()} for name, by_b in selections.items()},
        "count_one_hits": {name: per_unit(by_b[0.05] & count_one) for name, by_b in selections.items()},
        "strata": {name: {label: (int((by_b[0.05] & labels & (original >= lo) & (original <= hi)).sum()),
                                  int((labels & (original >= lo) & (original <= hi)).sum()))
                          for lo, hi, label in STRATA} for name, by_b in selections.items()},
        "average_precision": average_precision,
        "n": n,
        "original": original[labels],
    }


def analyze(tables: list[dict], bootstrap: int, seed: int) -> dict:
    units = sum((t["units"] for t in tables), [])
    positives = np.concatenate([t["positives"] for t in tables])
    count_one = np.concatenate([t["count_one"] for t in tables])
    names = list(tables[0]["counts"])
    counts = {name: {b: tuple(np.concatenate([t["counts"][name][b][i] for t in tables]) for i in (0, 1))
                     for b in FRACTIONS} for name in names}
    hits = {name: np.concatenate([t["count_one_hits"][name] for t in tables]) for name in names}
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.integers(0, len(units), len(units)) for _ in range(bootstrap)])

    def mean_f1(name: str, index: np.ndarray) -> np.ndarray:
        values = [2.0 * counts[name][b][1][index].sum(-1) / (counts[name][b][0][index].sum(-1) + positives[index].sum(-1))
                  for b in FRACTIONS]
        return np.mean(values, axis=0)

    def recall_one(name: str, index: np.ndarray) -> np.ndarray:
        return hits[name][index].sum(-1) / count_one[index].sum(-1)

    everyone = np.arange(len(units))
    rows = []
    for name in names:
        curve = [100 * 2.0 * counts[name][b][1].sum() / (counts[name][b][0].sum() + positives.sum()) for b in FRACTIONS]
        row = {"method": name,
               "average_precision": 100 * float(np.mean([t["average_precision"][name] for t in tables])),
               "f1_1": curve[0], "f1_5": curve[4], "f1_10": curve[9], "mean_f1_1_10": float(np.mean(curve))}
        for _, _, label in STRATA:
            hit = sum(t["strata"][name][label][0] for t in tables)
            total = sum(t["strata"][name][label][1] for t in tables)
            row[f"recall_at_5_count_{label}"] = 100 * hit / max(1, total)
        rows.append(row)

    comparisons = []
    references = [f"Safe Fusion, {design}" for design in DESIGNS]
    pairs = [(reference, other) for reference in references for other in names if other != reference]
    pairs += [(f"{selector}, mask-trained", f"{selector}, thinning-trained") for selector in SELECTORS if selector != "Safe Fusion"]
    for reference, other in pairs:
        for metric, function in [("mean_f1_1_10", mean_f1), ("count_one_recall_at_5", recall_one)]:
            point = float(function(reference, everyone) - function(other, everyone))
            sample = function(reference, draws) - function(other, draws)
            comparisons.append({"reference": reference, "comparator": other, "metric": metric,
                                "difference": 100 * point,
                                "low": 100 * float(np.quantile(sample, 0.025)),
                                "high": 100 * float(np.quantile(sample, 0.975))})
    original = np.concatenate([t["original"] for t in tables])
    n = sum(t["n"] for t in tables)
    composition = {
        "test_candidates": int(n), "positives": int(len(original)), "prevalence": float(len(original) / n),
        "positive_count_1_share": float(np.mean(original == 1)),
        "positive_count_2_5_share": float(np.mean((original >= 2) & (original <= 5))),
        "positive_count_gt5_share": float(np.mean(original > 5)),
        "bootstrap_units": len(units),
    }
    return {"composition": composition, "rows": rows, "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round2/thinning_transfer")
    parser.add_argument("--thinning-root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/thinning")
    parser.add_argument("--datasets", nargs="+", default=None, help="subset of dataset keys (default: all)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    specs = datasets(args.root, args.thinning_root)
    results, frames, differences = {}, [], []
    for key in args.datasets or list(specs):
        dataset = specs[key]
        data = ad.read_h5ad(dataset["data"] / "corrupted.h5ad")
        truth = ad.read_h5ad(dataset["truth"])
        if not np.array_equal(data.obs_names, truth.obs_names) or not np.array_equal(data.var_names, truth.var_names):
            raise ValueError(f"{key}: thinned and truth orders differ")
        thinned = dense(data.layers["corrupted_counts"]).astype(np.float32)
        truth_counts = dense(truth.layers["counts"]).astype(np.float32)
        cell_ids = data.obs_names.astype(str).to_numpy()
        bio_units = truth.obs[dataset["unit_column"]].astype(str).to_numpy()
        tables = [unit_table(dataset, unit, truth_counts, thinned, cell_ids, bio_units) for unit in dataset["units"]]
        results[key] = analyze(tables, args.bootstrap, args.seed)
        frame = pd.DataFrame(results[key]["rows"])
        frame.insert(0, "dataset", key)
        frames.append(frame)
        comparison = pd.DataFrame(results[key]["comparisons"])
        comparison.insert(0, "dataset", key)
        differences.append(comparison)
        print(f"== {key}: {results[key]['composition']}")
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(frame.drop(columns="dataset").round(2).to_string(index=False))
            print(comparison.drop(columns="dataset").round(2).to_string(index=False))
    output = args.root / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output / "transfer_summary.csv", index=False)
    pd.concat(differences, ignore_index=True).to_csv(output / "transfer_paired_differences.csv", index=False)
    (output / "transfer_summary.json").write_text(json.dumps(
        {"bootstrap": args.bootstrap, "seed": args.seed, "fractions": FRACTIONS, "results": results},
        indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
