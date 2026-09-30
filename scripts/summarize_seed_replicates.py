#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

EVIDENCE = Path("artifacts/paper_evidence")
PAN = Path("artifacts/pancreas_runs/0b2469810675-45c81b160d78")
COL = Path("artifacts/colon_runs/0b2469810675-c0db6f963e94")
CF = EVIDENCE / "pancreas_crossfit"
NORMAN = EVIDENCE / "norman_crispra"
REP = EVIDENCE / "seed_replicates"
GRID = np.linspace(0.001, 1.0, 1000)[:100]
INTEGER = np.arange(9, 100, 10)
COMPARATORS = ["SVD", "Weighted kNN", "MAGIC", "scVI"]
COUNT_SCALE = {
    "counts": lambda value, library: value,
    "normalized_expression_1e4": lambda value, library: value * (library / 1e4),
    "log1p_cpm": lambda value, library: np.expm1(value) * (library / 1e4),
}

def units(seed: int, evidence: Path = EVIDENCE, replicates: Path = REP) -> list[dict]:

    crossfit = evidence / "pancreas_crossfit"
    norman = evidence / "norman_crispra"
    production = seed == 1729
    out = []
    for fold in range(3):
        base = crossfit / f"fold_{fold}" if production else replicates / f"seed_{seed}" / f"pancreas_{fold}"
        data = PAN / "data/pancreas_islets" if production else replicates / f"seed_{seed}" / "data/pancreas"
        baselines = evidence / "baselines" if production else replicates / f"seed_{seed}" / "baselines"
        out.append(dict(
            dataset="Pancreas", tie_seed=seed + 100 * fold,
            corrupted=data / ("corrupted/mask_010.h5ad" if production else "corrupted.h5ad"),
            coordinates=data / ("coordinates/mask_010.parquet" if production else "coordinates.parquet"),
            splits=crossfit / f"fold_{fold}/splits.parquet",
            selector=base / ("selector_mlp_biology_range_fullteachers" if production else "selector"),
            contracts={"SVD": base / "svd_impute", "Weighted kNN": base / "graph_smooth",
                       "MAGIC": baselines / "magic/pancreas", "scVI": baselines / "scvi/pancreas"}))
    data = COL / "data/colon_epithelial" if production else replicates / f"seed_{seed}" / "data/colon"
    base = COL / "methods/standardized/colon_epithelial/mask_010" if production else replicates / f"seed_{seed}" / "colon"
    baselines = evidence / "baselines" if production else replicates / f"seed_{seed}" / "baselines"
    out.append(dict(
        dataset="Colon", tie_seed=seed + 1000,
        corrupted=data / ("corrupted/mask_010.h5ad" if production else "corrupted.h5ad"),
        coordinates=data / ("coordinates/mask_010.parquet" if production else "coordinates.parquet"),
        splits=COL / "data/colon_epithelial/splits.parquet",
        selector=evidence / "selector_mlp_biology_range/colon" if production else base / "selector",
        contracts={"SVD": base / "svd_impute", "Weighted kNN": base / "graph_smooth",
                   "MAGIC": baselines / "magic/colon", "scVI": baselines / "scvi/colon"}))
    data = norman if production else replicates / f"seed_{seed}" / "data/norman"
    base = norman / "methods" if production else replicates / f"seed_{seed}" / "norman"
    out.append(dict(
        dataset="CRISPRa", tie_seed=seed + 2000,
        corrupted=data / "corrupted.h5ad", coordinates=data / "coordinates.parquet",
        splits=norman / "splits.parquet",
        selector=evidence / "selector_mlp_biology_range_fullteachers/norman_crispra" if production else base / "selector",
        contracts={"SVD": base / "svd_impute", "Weighted kNN": base / "graph_smooth",
                   "MAGIC": baselines / "magic/norman", "scVI": baselines / "scvi/norman"}))
    return out

def unit_curves(unit: dict) -> dict[str, np.ndarray]:

    source = ad.read_h5ad(unit["corrupted"])
    matrix = source.layers["corrupted_counts"]
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    split = pd.read_parquet(unit["splits"]).set_index("cell_id").loc[source.obs_names.astype(str), "split"].to_numpy()
    test = np.flatnonzero(split == "test")
    local, cols = np.where(counts[test] == 0)
    rows = test[local]
    coordinates = pd.read_parquet(unit["coordinates"])
    masked = np.zeros(counts.shape, dtype=bool)
    masked[coordinates["cell_index"].to_numpy(int), coordinates["gene_index"].to_numpy(int)] = True
    labels = masked[rows, cols].astype(np.int8)
    library = counts.sum(axis=1)[rows]
    n, positives = len(labels), int(labels.sum())
    k = np.maximum(1, np.round(GRID * n).astype(np.int64))
    curves = {}
    for name, contract in unit["contracts"].items():
        metadata = json.loads((contract / "metadata.json").read_text())
        values = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols], dtype=np.float64)
        scores = COUNT_SCALE[metadata["scale"]](values, library)
        rng = np.random.default_rng(unit["tie_seed"])
        order = np.lexsort((rng.random(n), -np.nan_to_num(scores, nan=-np.inf)))
        cumulative = np.cumsum(labels[order], dtype=np.int64)
        curves[name] = np.stack([k, cumulative[k - 1], np.full(len(k), positives), np.full(len(k), n)], axis=1)
    report = json.loads((unit["selector"] / "calibration_report.json").read_text())
    if report["test"]["n_zeros"] != n or report["test"]["n_masked_positives"] != positives:
        raise ValueError(f"selector candidates differ from the evaluation candidates in {unit['selector']}")
    curve = report["exact_budget_curve"][:100]
    curves["Safe Fusion"] = np.array([[p["n_selected"], p["n_true_positive"], positives, n] for p in curve])
    return curves

def f1(pooled: np.ndarray) -> np.ndarray:
    selected, true_positive, positives = pooled[:, 0], pooled[:, 1], pooled[:, 2]
    return 100 * 2 * true_positive / (selected + positives)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 1730, 1731, 1732, 1733])
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE,
                        help="Root of the seed-1729 runs (pancreas_crossfit, norman_crispra, baselines, selectors).")
    parser.add_argument("--replicate-root", type=Path, default=None,
                        help="Root of the seed replicates. Defaults to <evidence-root>/seed_replicates.")
    parser.add_argument("--datasets", nargs="+", default=["Pancreas", "Colon", "CRISPRa"])
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <replicate-root>/summary.")
    args = parser.parse_args()
    replicate_root = args.replicate_root or args.evidence_root / "seed_replicates"
    args.output_dir = args.output_dir or replicate_root / "summary"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        pooled: dict[tuple[str, str], np.ndarray] = {}
        for unit in units(seed, args.evidence_root, replicate_root):
            if unit["dataset"] not in args.datasets:
                continue
            for name, curve in unit_curves(unit).items():
                key = (unit["dataset"], name)
                pooled[key] = curve if key not in pooled else pooled[key] + curve
        for (dataset, name), curve in pooled.items():
            values = f1(curve.astype(np.float64))
            for index, value in enumerate(values):
                rows.append(dict(seed=seed, dataset=dataset, method=name, fill_fraction=GRID[index], f1=value))
        print("finished seed", seed, flush=True)
    curves = pd.DataFrame(rows)
    curves.to_csv(args.output_dir / "seed_replicate_f1_curves.csv", index=False)

    integer = curves[np.isin(np.round(curves.fill_fraction * 1000).astype(int), INTEGER + 1)]
    mean_f1 = integer.groupby(["dataset", "seed", "method"]).f1.mean().unstack("method")
    summary = []
    for (dataset, seed), row in mean_f1.iterrows():
        grid = curves[(curves.dataset == dataset) & (curves.seed == seed)].pivot(index="fill_fraction", columns="method", values="f1")
        best_comparator = grid[COMPARATORS].max(axis=1)
        record = dict(dataset=dataset, seed=seed, **{f"mean_f1_{m}": row[m] for m in ["Safe Fusion", *COMPARATORS]},
                      grid_points_safe_fusion_best=int((grid["Safe Fusion"] >= best_comparator).sum()))
        for m in COMPARATORS:
            record[f"sf_minus_{m}"] = row["Safe Fusion"] - row[m]
        record["safe_fusion_best"] = bool(all(row["Safe Fusion"] > row[m] for m in COMPARATORS))
        summary.append(record)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output_dir / "seed_replicate_summary.csv", index=False)
    across = summary.groupby("dataset").agg(
        **{f"{c}_mean": (c, "mean") for c in summary.columns if c.startswith(("mean_f1_", "sf_minus_"))},
        **{f"{c}_sd": (c, "std") for c in summary.columns if c.startswith(("mean_f1_", "sf_minus_"))},
        replicates=("seed", "count"),
        replicates_safe_fusion_best=("safe_fusion_best", "sum"),
    )
    across.to_csv(args.output_dir / "seed_replicate_across.csv")
    pd.set_option("display.width", 250)
    print(summary.round(2).to_string(index=False))
    print(across.round(2).T.to_string())

if __name__ == "__main__":
    main()
