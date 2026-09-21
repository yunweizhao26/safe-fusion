#!/usr/bin/env python3
"""Evaluate unsupervised cell-type clustering on held-out biological units."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.downstream import (  # noqa: E402
    adjusted_rand_index,
    normalized_mutual_information,
    randomized_pca_embedding,
)


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_method(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("methods must use name=contract_directory")
    return name, Path(path)


def neighbor_purity_by_cell(embedding: np.ndarray, labels: np.ndarray, neighbors: int) -> np.ndarray:
    k = min(neighbors + 1, len(embedding))
    model = NearestNeighbors(n_neighbors=k).fit(embedding)
    indices = model.kneighbors(embedding, return_distance=False)
    indices = indices[:, 1:] if indices.shape[1] > 1 else indices[:, :0]
    if not indices.shape[1]:
        return np.full(len(embedding), np.nan)
    return np.mean(labels[indices] == labels[:, None], axis=1)


def bootstrap_summary(
    unit_seed_metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_unit = (
        unit_seed_metrics.groupby(["unit", "method", "metric"], observed=True)["value"]
        .mean()
        .reset_index()
    )
    units = sorted(per_unit["unit"].unique())
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(units), size=(replicates, len(units)))
    estimates: list[dict] = []
    comparisons: list[dict] = []
    for metric in sorted(per_unit["metric"].unique()):
        pivot = (
            per_unit[per_unit["metric"] == metric]
            .pivot(index="unit", columns="method", values="value")
            .reindex(units)
        )
        boot: dict[str, np.ndarray] = {}
        for method in sorted(pivot.columns):
            values = pivot[method].to_numpy(dtype=float)
            samples = np.nanmean(values[selections], axis=1)
            boot[method] = samples
            estimates.append({
                "method": method,
                "metric": metric,
                "estimate": float(np.nanmean(values)),
                "ci_low": float(np.nanquantile(samples, 0.025)),
                "ci_high": float(np.nanquantile(samples, 0.975)),
                "n_units": int(np.isfinite(values).sum()),
            })
        if "corrupted_raw" not in boot:
            continue
        for method in sorted(pivot.columns):
            if method in {"corrupted_raw", "reference_truth"}:
                continue
            difference = boot[method] - boot["corrupted_raw"]
            comparisons.append({
                "method": method,
                "reference": "corrupted_raw",
                "metric": metric,
                "difference": float(np.nanmean(pivot[method] - pivot["corrupted_raw"])),
                "ci_low": float(np.nanquantile(difference, 0.025)),
                "ci_high": float(np.nanquantile(difference, 0.975)),
                "n_units": int(np.isfinite(pivot[method] - pivot["corrupted_raw"]).sum()),
            })
    return pd.DataFrame(estimates), pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--unit-column", default="donor")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pca-components", type=int, default=30)
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--cluster-seeds", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    if truth.shape != corrupted.shape:
        raise ValueError("truth and corrupted shapes differ")
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        truth_adata.obs_names.astype(str), "split"
    ].to_numpy()
    training, test = split != "test", split == "test"
    if not training.any() or not test.any():
        raise ValueError("training and test cells are required")
    labels = truth_adata.obs[args.label_column].astype(str).to_numpy()
    units = truth_adata.obs[args.unit_column].astype(str).to_numpy()
    test_labels = labels[test]
    test_units = units[test]
    n_clusters = int(np.unique(test_labels).size)
    if n_clusters < 2:
        raise ValueError("at least two test cell types are required")

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
        passed = bool(
            parameters.get("test_used_for_fit") is False
            and parameters.get("fit_cells") == int(training.sum())
            and (
                not decision
                or decision.get("test_labels_used_for_thresholds") is False
            )
        )
        leakage.append({"method": name, "passed": passed})
    if not all(item["passed"] for item in leakage):
        raise ValueError("one or more method contracts failed leakage checks")

    embeddings = {
        name: randomized_pca_embedding(
            matrix,
            test,
            seed=args.seed,
            components=args.pca_components,
        )[test]
        for name, matrix in matrices.items()
    }
    records: list[dict] = []
    global_records: list[dict] = []
    test_unit_levels = sorted(np.unique(test_units))
    for seed_index in range(args.cluster_seeds):
        cluster_seed = args.seed + seed_index
        clusterings = {
            method: KMeans(n_clusters=n_clusters, n_init=1, random_state=cluster_seed).fit_predict(embedding)
            for method, embedding in embeddings.items()
        }
        reference_clusters = clusterings["reference_truth"]
        for method, predicted in clusterings.items():
            purity = neighbor_purity_by_cell(embeddings[method], test_labels, args.neighbors)
            global_records.extend([
                {"seed": cluster_seed, "method": method, "metric": "annotation_ari", "value": adjusted_rand_index(test_labels, predicted)},
                {"seed": cluster_seed, "method": method, "metric": "annotation_nmi", "value": normalized_mutual_information(test_labels, predicted)},
                {"seed": cluster_seed, "method": method, "metric": "reference_cluster_ari", "value": adjusted_rand_index(reference_clusters, predicted)},
                {"seed": cluster_seed, "method": method, "metric": "neighbor_purity", "value": float(np.nanmean(purity))},
            ])
            for unit in test_unit_levels:
                selected = test_units == unit
                if selected.sum() < 10 or np.unique(test_labels[selected]).size < 2:
                    continue
                records.extend([
                    {"unit": unit, "seed": cluster_seed, "method": method, "metric": "annotation_ari", "value": adjusted_rand_index(test_labels[selected], predicted[selected])},
                    {"unit": unit, "seed": cluster_seed, "method": method, "metric": "annotation_nmi", "value": normalized_mutual_information(test_labels[selected], predicted[selected])},
                    {"unit": unit, "seed": cluster_seed, "method": method, "metric": "reference_cluster_ari", "value": adjusted_rand_index(reference_clusters[selected], predicted[selected])},
                    {"unit": unit, "seed": cluster_seed, "method": method, "metric": "neighbor_purity", "value": float(np.nanmean(purity[selected]))},
                ])

    unit_seed_metrics = pd.DataFrame(records)
    summary, comparisons = bootstrap_summary(unit_seed_metrics, args.bootstrap, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit_seed_metrics.to_parquet(output / "unit_seed_metrics.parquet", index=False)
    pd.DataFrame(global_records).to_parquet(output / "global_seed_metrics.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    pd.DataFrame(leakage).to_parquet(output / "leakage_checks.parquet", index=False)
    report = {
        "design": "PCA and K-means fit without labels on held-out processed cells; imputation models fit on non-test cells",
        "cluster_count_source": "number of annotated cell types in the held-out set",
        "n_clusters": n_clusters,
        "n_training_cells": int(training.sum()),
        "n_test_cells": int(test.sum()),
        "n_test_units": int(len(test_unit_levels)),
        "cluster_seeds": args.cluster_seeds,
        "bootstrap_replicates": args.bootstrap,
        "methods": sorted(matrices),
        "leakage_checks_passed": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
