#!/usr/bin/env python3
"""Discover markers after imputation and validate them in independent donors.

Donors are divided into two deterministic folds.  For each method and cell
type, top positive markers are discovered in one fold from that method's
matrix, then evaluated against untouched reference counts in every donor of
the opposite fold.  The direction is swapped so every donor is used once for
validation.  This is distinct from truth-defined marker recovery: the marker
set itself is produced post facto by each imputed matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_method(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("methods must use name=matrix_or_contract_path")
    return name, Path(path)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if keep.sum() < 3 or np.ptp(left[keep]) == 0 or np.ptp(right[keep]) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(left[keep]), rankdata(right[keep]))[0, 1])


def effect(matrix: np.ndarray, labels: np.ndarray, rows: np.ndarray, label: str) -> np.ndarray:
    inside = rows[labels[rows] == label]
    outside = rows[labels[rows] != label]
    if len(inside) < 5 or len(outside) < 5:
        raise ValueError("insufficient cells for marker effect")
    return np.log1p(matrix[inside].mean(axis=0)) - np.log1p(matrix[outside].mean(axis=0))


def load_matrix(path: Path, full_n: int, subset: np.ndarray) -> np.ndarray:
    matrix_path = path / "mean.npy" if path.is_dir() else path
    matrix = np.load(matrix_path, allow_pickle=False, mmap_mode="r")
    if matrix.shape[0] == full_n:
        return np.asarray(matrix[subset], dtype=np.float32)
    if matrix.shape[0] == int(subset.sum()):
        return np.asarray(matrix, dtype=np.float32)
    raise ValueError(f"matrix row count does not match full or selected cells: {matrix_path}")


def donor_folds(
    donors: np.ndarray,
    strata: np.ndarray | None,
    seed: int,
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    unique_donors = np.unique(donors)
    assignments: dict[str, int] = {}
    if strata is None:
        shuffled = unique_donors.copy()
        rng.shuffle(shuffled)
        for index, donor in enumerate(shuffled):
            assignments[str(donor)] = index % 2
        return assignments
    donor_strata = {}
    for donor in unique_donors:
        values = np.unique(strata[donors == donor])
        if len(values) != 1:
            raise ValueError(f"donor {donor} spans multiple stratification values")
        donor_strata[str(donor)] = str(values[0])
    for value in sorted(set(donor_strata.values())):
        members = np.asarray(sorted(d for d, group in donor_strata.items() if group == value))
        rng.shuffle(members)
        for index, donor in enumerate(members):
            assignments[str(donor)] = index % 2
    return assignments


def summarize(
    donor_metrics: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = sorted(donor_metrics["method"].unique())
    metrics = sorted(donor_metrics["metric"].unique())
    donor_order = sorted(donor_metrics["donor"].unique())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(donor_order), size=(bootstrap, len(donor_order)))
    summaries = []
    comparisons = []
    for metric in metrics:
        pivot = (
            donor_metrics[donor_metrics["metric"].eq(metric)]
            .pivot(index="donor", columns="method", values="value")
            .reindex(donor_order)
        )
        samples = {}
        for method in methods:
            values = pivot[method].to_numpy(dtype=float)
            distribution = np.asarray([np.nanmean(values[index]) for index in indices])
            samples[method] = distribution
            summaries.append({
                "method": method,
                "metric": metric,
                "estimate": float(np.nanmean(values)),
                "ci_low": float(np.nanquantile(distribution, 0.025)),
                "ci_high": float(np.nanquantile(distribution, 0.975)),
                "n_donors": int(np.isfinite(values).sum()),
            })
        if "corrupted_raw" in samples:
            for method in methods:
                if method == "corrupted_raw":
                    continue
                difference = samples[method] - samples["corrupted_raw"]
                comparisons.append({
                    "method": method,
                    "reference": "corrupted_raw",
                    "metric": metric,
                    "difference": float(np.nanmean(pivot[method] - pivot["corrupted_raw"])),
                    "ci_low": float(np.nanquantile(difference, 0.025)),
                    "ci_high": float(np.nanquantile(difference, 0.975)),
                    "n_donors": int(np.isfinite(pivot[method] - pivot["corrupted_raw"]).sum()),
                })
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-splits", default=None)
    parser.add_argument("--subset-name", default="test")
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--donor-column", default="donor")
    parser.add_argument("--stratify-column", default=None)
    parser.add_argument("--marker-fraction", type=float, default=0.10)
    parser.add_argument("--min-markers", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    if not np.array_equal(truth_adata.obs_names.astype(str), corrupted_adata.obs_names.astype(str)):
        raise ValueError("truth and corrupted cell order differ")
    full_n = truth_adata.n_obs
    subset = np.ones(full_n, dtype=bool)
    if args.subset_splits:
        split = pd.read_parquet(args.subset_splits).set_index("cell_id").loc[
            truth_adata.obs_names.astype(str), "split"
        ].to_numpy()
        subset = split == args.subset_name

    truth = dense(truth_adata.layers["counts"])[subset].astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"])[subset].astype(np.float32)
    labels = truth_adata.obs[args.label_column].astype(str).to_numpy()[subset]
    donors = truth_adata.obs[args.donor_column].astype(str).to_numpy()[subset]
    strata = (
        truth_adata.obs[args.stratify_column].astype(str).to_numpy()[subset]
        if args.stratify_column else None
    )
    feature_names = (
        truth_adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in truth_adata.var else truth_adata.var_names.astype(str).to_numpy()
    )
    matrices = {"corrupted_raw": corrupted}
    for name, path in (parse_method(value) for value in args.method):
        matrices[name] = load_matrix(path, full_n, subset)
    if any(matrix.shape != truth.shape for matrix in matrices.values()):
        raise ValueError("one or more method matrices differ from the reference shape")

    assignments = donor_folds(donors, strata, args.seed)
    cell_fold = np.asarray([assignments[str(donor)] for donor in donors], dtype=int)
    marker_count = max(args.min_markers, int(round(args.marker_fraction * truth.shape[1])))
    marker_count = min(marker_count, truth.shape[1])
    discovered_rows = []
    donor_rows = []

    for discovery_fold in (0, 1):
        discovery = np.flatnonzero(cell_fold == discovery_fold)
        validation_donors = sorted(set(donors[cell_fold != discovery_fold]))
        for method, matrix in matrices.items():
            for label in sorted(np.unique(labels[discovery])):
                try:
                    discovery_effect = effect(matrix, labels, discovery, label)
                except ValueError:
                    continue
                selected = np.argsort(discovery_effect)[::-1][:marker_count]
                for rank, gene in enumerate(selected, start=1):
                    discovered_rows.append({
                        "discovery_fold": discovery_fold,
                        "method": method,
                        "cell_type": label,
                        "rank": rank,
                        "gene_id": str(feature_names[gene]),
                        "discovery_effect": float(discovery_effect[gene]),
                    })
                for donor in validation_donors:
                    rows = np.flatnonzero(donors == donor)
                    try:
                        reference_effect = effect(truth, labels, rows, label)
                    except ValueError:
                        continue
                    reference_top = set(np.argsort(reference_effect)[::-1][:marker_count].tolist())
                    metrics = {
                        "replication_precision_at_k": float(np.mean([gene in reference_top for gene in selected])),
                        "reference_positive_fraction": float(np.mean(reference_effect[selected] > 0)),
                        "reference_effect_mean": float(np.mean(reference_effect[selected])),
                        "all_gene_effect_spearman": spearman(discovery_effect, reference_effect),
                    }
                    for metric, value in metrics.items():
                        donor_rows.append({
                            "donor": donor,
                            "discovery_fold": discovery_fold,
                            "method": method,
                            "cell_type": label,
                            "metric": metric,
                            "value": value,
                        })

    detailed = pd.DataFrame(donor_rows)
    donor_metrics = (
        detailed.groupby(["donor", "discovery_fold", "method", "metric"], as_index=False)["value"]
        .mean()
    )
    summary, comparisons = summarize(donor_metrics, args.bootstrap, args.seed + 1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(discovered_rows).to_parquet(output / "discovered_markers.parquet", index=False)
    detailed.to_parquet(output / "celltype_donor_metrics.parquet", index=False)
    donor_metrics.to_parquet(output / "donor_metrics.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    report = {
        "design": "two-fold donor marker discovery with opposite-fold reference-count validation",
        "n_donors": int(len(np.unique(donors))),
        "n_cells": int(len(donors)),
        "marker_count_per_cell_type": int(marker_count),
        "methods": sorted(matrices),
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        "donor_fold_assignments": assignments,
        "artifacts": [
            "discovered_markers.parquet", "celltype_donor_metrics.parquet",
            "donor_metrics.parquet", "bootstrap_summary.parquet",
            "paired_comparisons.parquet",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
