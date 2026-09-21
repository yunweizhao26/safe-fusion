#!/usr/bin/env python3
"""Validate regulatory-response edges with real gain- or loss-of-function data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import roc_auc_score


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.metrics import average_precision_tie_aware, spearman  # noqa: E402


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_method(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("methods must use name=contract_directory")
    return name, Path(path)


def log2fc(matrix: np.ndarray, condition: np.ndarray, control: np.ndarray) -> np.ndarray:
    return np.log2(matrix[condition].mean(axis=0) + 1.0) - np.log2(
        matrix[control].mean(axis=0) + 1.0
    )


def robust_edge_labels(
    development_effect: np.ndarray,
    first_half_effect: np.ndarray,
    second_half_effect: np.ndarray,
    source_index: int,
    fraction: float,
) -> np.ndarray:
    eligible = np.ones(len(development_effect), dtype=bool)
    eligible[source_index] = False
    threshold = np.quantile(np.abs(development_effect[eligible]), 1.0 - fraction)
    consistent = (
        (np.sign(first_half_effect) == np.sign(second_half_effect))
        & (np.sign(first_half_effect) == np.sign(development_effect))
    )
    labels = eligible & consistent & (np.abs(development_effect) >= threshold)
    return labels


def jaccard(first: np.ndarray, second: np.ndarray) -> float:
    union = np.sum(first | second)
    return float(np.sum(first & second) / union) if union else 1.0


def binary_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def summarize_units(
    unit_metrics: pd.DataFrame,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = sorted(unit_metrics["unit"].unique())
    rng = np.random.default_rng(seed)
    selections = rng.integers(0, len(units), size=(replicates, len(units)))
    estimates: list[dict] = []
    comparisons: list[dict] = []
    for metric in sorted(unit_metrics["metric"].unique()):
        pivot = (
            unit_metrics[unit_metrics["metric"] == metric]
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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--intervention", choices=["gain_of_function", "loss_of_function"], required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--min-cells-per-split", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--publication-doi", default="")
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        truth_adata.obs_names.astype(str), "split"
    ].to_numpy()
    development, test = split == "development", split == "test"
    targets = truth_adata.obs["target"].astype(str).to_numpy()
    conditions = truth_adata.obs["condition"].astype(str).to_numpy()
    controls = (targets == "none") | (conditions == "ctrl")
    genes = (
        truth_adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in truth_adata.var
        else truth_adata.var_names.astype(str).to_numpy()
    )
    gene_lookup = {gene: index for index, gene in enumerate(genes)}

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
            and parameters.get("fit_cells") == int(development.sum())
            and (not decision or decision.get("test_labels_used_for_thresholds") is False)
        )
        leakage.append({"method": name, "passed": passed})
    if not all(item["passed"] for item in leakage):
        raise ValueError("one or more method contracts failed leakage checks")

    control_development = np.flatnonzero(development & controls)
    control_test = np.flatnonzero(test & controls)
    if len(control_development) < 20 or len(control_test) < 20:
        raise ValueError("at least twenty control cells are required in each split")
    control_halves = np.array_split(control_development, 2)
    unit_rows: list[dict] = []
    edge_rows: list[dict] = []
    retained_targets: list[str] = []
    for target in sorted(set(targets) - {"none"}):
        if target not in gene_lookup:
            continue
        dev_condition = np.flatnonzero(development & (targets == target))
        test_condition = np.flatnonzero(test & (targets == target))
        if (
            len(dev_condition) < args.min_cells_per_split
            or len(test_condition) < args.min_cells_per_split
        ):
            continue
        condition_halves = np.array_split(dev_condition, 2)
        source_index = gene_lookup[target]
        development_effect = log2fc(truth, dev_condition, control_development)
        first_half_effect = log2fc(truth, condition_halves[0], control_halves[0])
        second_half_effect = log2fc(truth, condition_halves[1], control_halves[1])
        labels_05 = robust_edge_labels(
            development_effect, first_half_effect, second_half_effect, source_index, 0.05
        )
        labels_10 = robust_edge_labels(
            development_effect, first_half_effect, second_half_effect, source_index, 0.10
        )
        if labels_10.sum() < 10:
            continue
        retained_targets.append(target)
        edge_rows.extend(
            {
                "regulator": target,
                "target_gene": gene,
                "development_log2fc": float(development_effect[index]),
                "first_half_log2fc": float(first_half_effect[index]),
                "second_half_log2fc": float(second_half_effect[index]),
                "robust_edge_q05": bool(labels_05[index]),
                "robust_edge_q10": bool(labels_10[index]),
            }
            for index, gene in enumerate(genes)
            if index != source_index
        )
        test_truth_effect = log2fc(truth, test_condition, control_test)
        eligible = np.ones(len(genes), dtype=bool)
        eligible[source_index] = False
        top10_threshold = np.quantile(np.abs(development_effect[eligible]), 0.90)
        development_top10 = eligible & (np.abs(development_effect) >= top10_threshold)
        for method, matrix in matrices.items():
            predicted = log2fc(matrix, test_condition, control_test)
            score = np.abs(predicted)
            predicted_top10_threshold = np.quantile(score[eligible], 0.90)
            predicted_top10 = eligible & (score >= predicted_top10_threshold)
            source_error = abs(float(predicted[source_index] - test_truth_effect[source_index]))
            metrics = {
                "edge_pr_auc_q05": average_precision_tie_aware(labels_05[eligible], score[eligible]),
                "edge_pr_auc_q10": average_precision_tie_aware(labels_10[eligible], score[eligible]),
                "edge_roc_auc_q05": binary_roc_auc(labels_05[eligible], score[eligible]),
                "edge_roc_auc_q10": binary_roc_auc(labels_10[eligible], score[eligible]),
                "effect_spearman_development": spearman(development_effect[eligible], predicted[eligible]),
                "effect_spearman_test_truth": spearman(test_truth_effect[eligible], predicted[eligible]),
                "effect_rmse_test_truth": float(np.sqrt(np.mean(np.square(predicted[eligible] - test_truth_effect[eligible])))),
                "direction_accuracy_q10": float(np.mean(np.sign(predicted[labels_10]) == np.sign(development_effect[labels_10]))),
                "top10_edge_jaccard": jaccard(development_top10, predicted_top10),
                "source_log2fc_abs_error": source_error,
            }
            unit_rows.extend(
                {"unit": target, "method": method, "metric": metric, "value": value}
                for metric, value in metrics.items()
            )

    unit_metrics = pd.DataFrame(unit_rows)
    if unit_metrics.empty:
        raise RuntimeError("no perturbation units passed the real-GRN gates")
    summary, comparisons = summarize_units(unit_metrics, args.bootstrap, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit_metrics.to_parquet(output / "unit_metrics.parquet", index=False)
    pd.DataFrame(edge_rows).to_parquet(output / "development_edges.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    pd.DataFrame(leakage).to_parquet(output / "leakage_checks.parquet", index=False)
    report = {
        "dataset": args.dataset,
        "intervention": args.intervention,
        "publication_doi": args.publication_doi,
        "design": "development-cell interventions define robust response edges; held-out cells evaluate edge recovery",
        "edge_definition": "top 5 or 10 percent absolute development log2 fold changes with matching signs in two development halves",
        "edge_scope": "causal perturbation-response edges, not necessarily direct molecular binding",
        "source_gene_excluded_from_edge_metrics": True,
        "n_regulators": int(len(retained_targets)),
        "regulators": retained_targets,
        "n_development_cells": int(development.sum()),
        "n_test_cells": int(test.sum()),
        "bootstrap_unit": "perturbed regulator",
        "bootstrap_replicates": args.bootstrap,
        "minimum_cells_per_condition_split": args.min_cells_per_split,
        "replicate_limitation": "source objects lack independent biological replicate fields",
        "methods": sorted(matrices),
        "leakage_checks_passed": True,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
