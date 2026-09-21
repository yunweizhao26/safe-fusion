#!/usr/bin/env python3
"""Evaluate developmental-order and lineage preservation on held-out cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def log1p_cpm(counts: np.ndarray) -> np.ndarray:
    library = counts.sum(axis=1, keepdims=True)
    scale = np.divide(10_000.0, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(counts * scale).astype(np.float32)


def parse_method(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    value = spearmanr(first, second).statistic
    return float(value) if np.isfinite(value) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        truth_adata.obs_names.astype(str), "split"
    ].to_numpy()
    development, test = split == "development", split == "test"
    stage = truth_adata.obs["stage_rank"].to_numpy(dtype=int)
    lineage = truth_adata.obs["lineages"].astype(str).to_numpy()
    stage_levels = np.sort(np.unique(stage))

    learned = {name: path for name, path in (parse_method(value) for value in args.method)}
    matrices = {
        "reference_truth": truth,
        "corrupted_raw": corrupted,
        **{name: np.load(path / "mean.npy", allow_pickle=False) for name, path in learned.items()},
    }
    leakage: list[dict] = []
    for name, path in learned.items():
        metadata = json.loads((path / "metadata.json").read_text())
        parameters = metadata.get("parameters", {})
        decision = parameters.get("decision_layer", {})
        decision_passed = (
            not decision
            or decision.get("test_values_used_for_thresholds") is False
            or (
                decision.get("budget_mode") == "apply_topk"
                and decision.get("test_labels_used_for_thresholds") is False
            )
        )
        passed = bool(
            parameters.get("test_used_for_fit") is False
            and parameters.get("fit_cells") == int(development.sum())
            and decision_passed
        )
        leakage.append({"method": name, "passed": passed})
    if not all(item["passed"] for item in leakage):
        raise ValueError("method failed leakage audit")

    truth_normalized = log1p_cpm(truth)
    corrupted_normalized = log1p_cpm(corrupted)
    dev_stage_means = np.vstack([truth_normalized[development & (stage == value)].mean(axis=0) for value in stage_levels])
    dynamic_score = np.asarray([abs(correlation(stage_levels, dev_stage_means[:, gene])) for gene in range(truth.shape[1])])
    dynamic = np.argsort(dynamic_score)[-max(20, truth.shape[1] // 10):]
    components = min(30, int(development.sum()) - 1, corrupted_normalized.shape[1] - 1)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=args.seed)
    development_embedding = pca.fit_transform(corrupted_normalized[development])
    time_model = KNeighborsRegressor(n_neighbors=30, weights="distance").fit(
        development_embedding, stage[development]
    )
    lineage_model = KNeighborsClassifier(n_neighbors=30, weights="distance").fit(
        development_embedding, lineage[development]
    )
    neighbor_model = NearestNeighbors(n_neighbors=30).fit(development_embedding)
    records: list[dict] = []
    overall: list[dict] = []
    for method, matrix in matrices.items():
        normalized = log1p_cpm(matrix)
        test_embedding = pca.transform(normalized[test])
        predicted_time = time_model.predict(test_embedding)
        predicted_lineage = lineage_model.predict(test_embedding)
        neighbor_indices = neighbor_model.kneighbors(test_embedding, return_distance=False)
        neighbor_stages = stage[development][neighbor_indices]
        for value in stage_levels:
            local_test = stage[test] == value
            local = np.flatnonzero(test & (stage == value))
            if not local_test.any():
                continue
            method_mean = normalized[local].mean(axis=0)
            truth_mean = truth_normalized[local].mean(axis=0)
            records.extend([
                {"unit": int(value), "method": method, "metric": "stage_rank_mae", "value": float(np.mean(np.abs(predicted_time[local_test] - value)))},
                {"unit": int(value), "method": method, "metric": "lineage_accuracy", "value": float(np.mean(predicted_lineage[local_test] == lineage[local]))},
                {"unit": int(value), "method": method, "metric": "adjacent_stage_neighbor_rate", "value": float(np.mean(np.abs(neighbor_stages[local_test] - value) <= 1))},
                {"unit": int(value), "method": method, "metric": "stage_pseudobulk_spearman", "value": correlation(truth_mean, method_mean)},
                {"unit": int(value), "method": method, "metric": "dynamic_gene_spearman", "value": correlation(truth_mean[dynamic], method_mean[dynamic])},
            ])
        overall.append({
            "method": method,
            "test_stage_spearman": correlation(stage[test], predicted_time),
            "test_stage_mae": float(np.mean(np.abs(predicted_time - stage[test]))),
            "test_lineage_balanced_accuracy": float(balanced_accuracy_score(lineage[test], predicted_lineage)),
        })

    unit_metrics = pd.DataFrame(records)
    estimates: list[dict] = []
    comparisons: list[dict] = []
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, len(stage_levels), size=(args.bootstrap, len(stage_levels))
    )
    for metric in sorted(unit_metrics["metric"].unique()):
        pivot = (
            unit_metrics[unit_metrics["metric"] == metric]
            .pivot(index="unit", columns="method", values="value")
            .reindex(stage_levels)
        )
        samples_by_method: dict[str, np.ndarray] = {}
        for method in sorted(pivot.columns):
            values = pivot[method].to_numpy(dtype=float)
            samples = np.mean(values[bootstrap_indices], axis=1)
            samples_by_method[method] = samples
            estimates.append({
                "method": method,
                "metric": metric,
                "estimate": float(values.mean()),
                "ci_low": float(np.quantile(samples, 0.025)),
                "ci_high": float(np.quantile(samples, 0.975)),
                "n_stages": int(len(values)),
            })
        raw = pivot["corrupted_raw"].to_numpy(dtype=float)
        for method in sorted(pivot.columns):
            if method in {"corrupted_raw", "reference_truth"}:
                continue
            difference = pivot[method].to_numpy(dtype=float) - raw
            bootstrap_difference = (
                samples_by_method[method] - samples_by_method["corrupted_raw"]
            )
            comparisons.append({
                "method": method,
                "reference": "corrupted_raw",
                "metric": metric,
                "difference": float(difference.mean()),
                "ci_low": float(np.quantile(bootstrap_difference, 0.025)),
                "ci_high": float(np.quantile(bootstrap_difference, 0.975)),
                "n_stages": int(len(difference)),
            })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit_metrics.to_parquet(output / "unit_metrics.parquet", index=False)
    pd.DataFrame(estimates).to_parquet(output / "bootstrap_summary.parquet", index=False)
    pd.DataFrame(comparisons).to_parquet(output / "paired_comparisons.parquet", index=False)
    pd.DataFrame(overall).to_parquet(output / "overall_metrics.parquet", index=False)
    pd.DataFrame(leakage).to_parquet(output / "leakage_checks.parquet", index=False)
    report = {
        "design": "held-out cells within each developmental stage; a common PCA and downstream model are fit on corrupted development cells, and each processed test matrix is transformed without refitting",
        "n_stages": int(len(stage_levels)),
        "n_test_cells": int(test.sum()),
        "dynamic_genes_defined_on_development": int(len(dynamic)),
        "methods": sorted(matrices),
        "leakage_checks_passed": True,
        "limitation": "processed object has no embryo replicate; stages are bootstrap units",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
