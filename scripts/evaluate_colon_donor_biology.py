#!/usr/bin/env python3
"""Donor-level colon biology evaluation on the locked test donors with bootstrap inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cdist


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.downstream import randomized_pca_embedding  # noqa: E402
from safefusion_benchmark.metrics import average_precision_tie_aware, log1p_mae, spearman  # noqa: E402


CELL_MARKERS = {
    "BEST4": "Enterocytes BEST4", "OTOP2": "Enterocytes BEST4",
    "CA7": "Enterocytes BEST4", "GUCA2A": "Enterocytes BEST4", "GUCA2B": "Enterocytes BEST4",
    "CA1": "Enterocytes CA1", "CA2": "Enterocytes CA1",
    "TMIGD1": "Enterocytes TMIGD1", "MEP1A": "Enterocytes TMIGD1",
    "MUC2": "Goblet", "TFF3": "Goblet", "AGR2": "Goblet", "SPDEF": "Goblet",
    "TFF1": "Goblet cells MUC2 TFF1", "SPINK4": "Goblet cells SPINK4",
    "POU2F3": "Tuft", "DCLK1": "Tuft", "CHGA": "Enteroendocrine",
    "GCG": "Enteroendocrine", "GIP": "Enteroendocrine", "CCK": "Enteroendocrine",
    "LYZ": "Paneth", "DEFA5": "Paneth", "MKI67": "Cycling", "PCNA": "Cycling",
    "OLFM4": "Stem", "LGR5": "Stem", "ASCL2": "Stem", "SOX9": "Stem",
}

INFLAMMATION_MARKERS = {
    "REG1A", "REG3A", "DUOX2", "NOS2", "CXCL1", "CXCL2", "CXCL3", "CXCL8",
    "CCL20", "IL32", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "HLA-A", "HLA-B",
    "STAT1", "IRF1", "IFITM1", "IFITM3",
}

SELECTED_MARKERS = sorted(set(CELL_MARKERS) | INFLAMMATION_MARKERS)


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_method(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("methods must use name=contract_directory")
    return name, Path(path)


def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    values: list[float] = []
    for label in np.unique(actual):
        tp = np.sum((actual == label) & (predicted == label))
        fp = np.sum((actual != label) & (predicted == label))
        fn = np.sum((actual == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        values.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(values))


def centroid_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> np.ndarray:
    embedding = randomized_pca_embedding(matrix, train, seed=seed, components=30)
    classes = np.unique(labels[train])
    centroids = np.stack([embedding[train & (labels == label)].mean(axis=0) for label in classes])
    return classes[np.argmin(cdist(embedding[test], centroids), axis=1)]


def marker_target(labels: np.ndarray, target: str) -> np.ndarray:
    return np.char.find(labels.astype(str), target) >= 0


def canonical_marker_metrics(
    matrix: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    gene_lookup: dict[str, int],
) -> dict[str, float]:
    average_precisions: list[float] = []
    true_effects: list[float] = []
    predicted_effects: list[float] = []
    ectopic_filled = 0
    ectopic_total = 0
    for marker in SELECTED_MARKERS:
        if marker not in gene_lookup:
            continue
        gene = gene_lookup[marker]
        target = marker_target(labels[rows], CELL_MARKERS[marker]) if marker in CELL_MARKERS else np.zeros(len(rows), dtype=bool)
        if not target.any() or target.all():
            continue
        true_values = truth[rows, gene]
        predicted_values = matrix[rows, gene]
        average_precisions.append(average_precision_tie_aware(target, predicted_values))
        true_effects.append(float(np.log1p(true_values[target].mean()) - np.log1p(true_values[~target].mean())))
        predicted_effects.append(float(np.log1p(predicted_values[target].mean()) - np.log1p(predicted_values[~target].mean())))
        ectopic = (~target) & (true_values == 0)
        ectopic_filled += int(np.sum(predicted_values[ectopic] > 0.5))
        ectopic_total += int(np.sum(ectopic))
    return {
        "canonical_marker_pr_auc": float(np.mean(average_precisions)) if average_precisions else float("nan"),
        "canonical_marker_effect_spearman": spearman(np.asarray(true_effects), np.asarray(predicted_effects)),
        "ectopic_marker_fill_rate": float(ectopic_filled / ectopic_total) if ectopic_total else float("nan"),
    }


def development_marker_metrics(
    matrix: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float]:
    average_precisions: list[float] = []
    correlations: list[float] = []
    for label in np.unique(labels[train]):
        in_train = train & (labels == label)
        out_train = train & (labels != label)
        target = labels[rows] == label
        if not in_train.any() or not out_train.any() or not target.any() or target.all():
            continue
        reference_effect = np.log1p(truth[in_train].mean(axis=0)) - np.log1p(truth[out_train].mean(axis=0))
        markers = reference_effect >= np.quantile(reference_effect, 0.9)
        predicted_effect = np.log1p(matrix[rows[target]].mean(axis=0)) - np.log1p(matrix[rows[~target]].mean(axis=0))
        average_precisions.append(average_precision_tie_aware(markers, predicted_effect))
        correlations.append(spearman(reference_effect, predicted_effect))
    return {
        "development_marker_pr_auc": float(np.mean(average_precisions)),
        "development_marker_rank_spearman": float(np.mean(correlations)),
    }


def bootstrap_indices(donors: np.ndarray, replicates: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, len(donors), size=len(donors)) for _ in range(replicates)]


def summarize_unit_metrics(
    unit_metrics: pd.DataFrame,
    bootstrap_indices: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates: list[dict] = []
    comparisons: list[dict] = []
    methods = sorted(unit_metrics["method"].unique())
    metrics = sorted(unit_metrics["metric"].unique())
    donor_order = unit_metrics[["donor", "disease"]].drop_duplicates().sort_values("donor")
    for metric in metrics:
        pivot = (
            unit_metrics[unit_metrics["metric"] == metric]
            .pivot(index="donor", columns="method", values="value")
            .reindex(donor_order["donor"])
        )
        bootstrap_values: dict[str, np.ndarray] = {}
        for method in methods:
            values = pivot[method].to_numpy(dtype=float)
            samples = np.asarray([np.nanmean(values[index]) for index in bootstrap_indices])
            bootstrap_values[method] = samples
            estimates.append({
                "scope": "donor",
                "method": method,
                "metric": metric,
                "estimate": float(np.nanmean(values)),
                "ci_low": float(np.nanquantile(samples, 0.025)),
                "ci_high": float(np.nanquantile(samples, 0.975)),
                "n_units": int(np.sum(np.isfinite(values))),
            })
        for method in methods:
            if method in {"reference_truth", "corrupted_raw"}:
                continue
            for reference in ("corrupted_raw", "graph_smooth"):
                if reference not in bootstrap_values or method == reference:
                    continue
                difference = bootstrap_values[method] - bootstrap_values[reference]
                observed = float(np.nanmean(pivot[method] - pivot[reference]))
                comparisons.append({
                    "scope": "donor",
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "difference": observed,
                    "ci_low": float(np.nanquantile(difference, 0.025)),
                    "ci_high": float(np.nanquantile(difference, 0.975)),
                    "n_units": int(np.sum(np.isfinite(pivot[method] - pivot[reference]))),
                })
    return pd.DataFrame(estimates), pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", required=True, help="name=contract_directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    labels = truth_adata.obs["cell_type"].astype(str).to_numpy()
    broad_labels = truth_adata.obs["cell_type_broad"].astype(str).to_numpy()
    donors = truth_adata.obs["donor"].astype(str).to_numpy()
    diseases = truth_adata.obs["disease"].astype(str).to_numpy()
    feature_names = truth_adata.var["feature_name"].astype(str).to_numpy()
    gene_lookup = {gene: index for index, gene in enumerate(feature_names)}
    split_frame = pd.read_parquet(args.splits).set_index("cell_id").loc[truth_adata.obs_names.astype(str)]
    split = split_frame["split"].to_numpy()
    train = np.isin(split, ["development", "validation"])
    test = split == "test"
    if not train.any() or not test.any():
        raise ValueError("both training and locked test splits are required")

    learned = {name: path for name, path in (parse_method(value) for value in args.method)}
    matrices: dict[str, np.ndarray] = {
        "reference_truth": truth,
        "corrupted_raw": corrupted,
        **{name: np.load(path / "mean.npy", allow_pickle=False) for name, path in learned.items()},
    }

    leakage_checks: list[dict] = []
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
        leakage_checks.append({
            "method": name,
            "training_cells": parameters.get("fit_cells"),
            "heldout_cells": parameters.get("heldout_test_cells"),
            "test_used_for_fit": parameters.get("test_used_for_fit"),
            "passed": bool(
                parameters.get("test_used_for_fit") is False
                and parameters.get("fit_cells") == int(train.sum())
                and decision_passed
            ),
        })
    if not all(item["passed"] for item in leakage_checks):
        raise ValueError("one or more method contracts failed the leakage audit")

    coordinates = pd.read_parquet(args.coordinates, filters=[("split", "==", "test")])
    coordinate_rows = coordinates["cell_index"].to_numpy(dtype=int)
    coordinate_genes = coordinates["gene_index"].to_numpy(dtype=int)
    coordinate_symbols = feature_names[coordinate_genes]
    marker_coordinates = coordinates[np.isin(coordinate_symbols, SELECTED_MARKERS)].copy()
    marker_coordinate_rows = marker_coordinates["cell_index"].to_numpy(dtype=int)
    marker_coordinate_genes = marker_coordinates["gene_index"].to_numpy(dtype=int)
    marker_original = marker_coordinates["original_value"].to_numpy(dtype=float)

    donor_records: list[dict] = []
    test_donors = sorted(set(donors[test]))
    for method, matrix in matrices.items():
        detailed_prediction = centroid_predictions(matrix, labels, train, test, args.seed)
        broad_prediction = centroid_predictions(matrix, broad_labels, train, test, args.seed + 100)
        for donor in test_donors:
            rows = np.flatnonzero(test & (donors == donor))
            local_test = np.flatnonzero(np.isin(np.flatnonzero(test), rows))
            donor_coordinates = marker_coordinates[np.isin(marker_coordinate_rows, rows)]
            if len(donor_coordinates):
                masked_mae = log1p_mae(
                    marker_original[np.isin(marker_coordinate_rows, rows)],
                    matrix[donor_coordinates["cell_index"].to_numpy(dtype=int), donor_coordinates["gene_index"].to_numpy(dtype=int)],
                )
            else:
                masked_mae = float("nan")
            metrics = {
                "cell_identity_macro_f1": macro_f1(labels[rows], detailed_prediction[local_test]),
                "broad_cell_identity_macro_f1": macro_f1(broad_labels[rows], broad_prediction[local_test]),
                "masked_canonical_marker_log1p_mae": masked_mae,
                **canonical_marker_metrics(matrix, truth, labels, rows, gene_lookup),
                **development_marker_metrics(matrix, truth, labels, train, rows),
            }
            for metric, value in metrics.items():
                donor_records.append({
                    "donor": donor,
                    "disease": diseases[rows[0]],
                    "method": method,
                    "metric": metric,
                    "value": value,
                    "n_cells": int(len(rows)),
                })

    unit_metrics = pd.DataFrame(donor_records)
    donor_frame = unit_metrics[["donor", "disease"]].drop_duplicates().sort_values("donor")
    indices = bootstrap_indices(donor_frame["donor"].to_numpy(), args.bootstrap, args.seed)
    summary, comparisons = summarize_unit_metrics(unit_metrics, indices)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit_metrics.to_parquet(output / "donor_metrics.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    pd.DataFrame(leakage_checks).to_parquet(output / "leakage_checks.parquet", index=False)
    for method, matrix in matrices.items():
        np.save(output / f"test_{method}.npy", matrix[test], allow_pickle=False)
    report = {
        "design": "locked donor split; 17 development / 8 validation / 9 test donors",
        "biological_unit": "donor",
        "test_donors": test_donors,
        "bootstrap_replicates": args.bootstrap,
        "methods": sorted(unit_metrics["method"].unique()),
        "leakage_checks_passed": True,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
