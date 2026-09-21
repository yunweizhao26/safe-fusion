#!/usr/bin/env python3
"""Perturbation-preservation evaluation on locked test cells with bootstrap over perturbations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.metrics import average_precision_tie_aware, log1p_mae, spearman  # noqa: E402


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_method(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("methods must use name=contract_directory")
    return name, Path(path)


def log2fc_vector(reference: np.ndarray, condition: np.ndarray, control: np.ndarray) -> np.ndarray:
    return np.log2(np.mean(condition, axis=0) + 1.0) - np.log2(np.mean(control, axis=0) + 1.0)


def bootstrap_indices(n_units: int, replicates: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n_units, size=n_units) for _ in range(replicates)]


def summarize_units(
    unit_metrics: pd.DataFrame,
    indices: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates: list[dict] = []
    comparisons: list[dict] = []
    methods = sorted(unit_metrics["method"].unique())
    metrics = sorted(unit_metrics["metric"].unique())
    unit_order = np.sort(unit_metrics["unit"].unique())
    for metric in metrics:
        pivot = (
            unit_metrics[unit_metrics["metric"] == metric]
            .pivot(index="unit", columns="method", values="value")
            .reindex(unit_order)
        )
        bootstrap_values: dict[str, np.ndarray] = {}
        for method in methods:
            values = pivot[method].to_numpy(dtype=float)
            samples = np.asarray([np.nanmean(values[index]) for index in indices])
            bootstrap_values[method] = samples
            estimates.append({
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
    conditions = np.asarray(truth_adata.obs["condition"].astype(str).to_numpy(), dtype=str)
    targets = np.asarray(truth_adata.obs["target"].astype(str).to_numpy(), dtype=str)
    controls = conditions == "ctrl"
    feature_names = (
        truth_adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in truth_adata.var
        else np.asarray(truth_adata.var_names, dtype=str)
    )
    split_frame = pd.read_parquet(args.splits).set_index("cell_id").loc[truth_adata.obs_names.astype(str)]
    split = split_frame["split"].to_numpy()
    development = split == "development"
    test = split == "test"

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
                and parameters.get("fit_cells") == int(development.sum())
                and decision_passed
            ),
        })
    if not all(item["passed"] for item in leakage_checks):
        raise ValueError("one or more method contracts failed the leakage audit")

    condition_levels = [value for value in sorted(set(targets)) if value != "none"]
    gene_lookup = {gene: index for index, gene in enumerate(feature_names)}
    unit_records: list[dict] = []
    control_metrics: list[dict] = []
    for target in condition_levels:
        condition_cells = np.flatnonzero(test & (targets == target))
        control_cells = np.flatnonzero(test & controls)
        dev_condition = np.flatnonzero(development & (targets == target))
        dev_control = np.flatnonzero(development & controls)
        if len(condition_cells) < 10 or len(control_cells) < 10 or len(dev_condition) < 10 or len(dev_control) < 10:
            continue
        if target not in gene_lookup:
            continue
        target_gene = gene_lookup[target]
        reference_logfc = log2fc_vector(truth, truth[condition_cells], truth[control_cells])
        development_reference = log2fc_vector(truth, truth[dev_condition], truth[dev_control])
        response = np.abs(development_reference) >= np.quantile(np.abs(development_reference), 0.9)
        strong = np.abs(reference_logfc) >= np.quantile(np.abs(reference_logfc), 0.9)
        for method, matrix in matrices.items():
            predicted_logfc = log2fc_vector(matrix, matrix[condition_cells], matrix[control_cells])
            target_predicted = float(predicted_logfc[target_gene])
            target_reference = float(reference_logfc[target_gene])
            condition_truth_target = truth[condition_cells, target_gene]
            condition_pred_target = matrix[condition_cells, target_gene]
            control_truth_target = truth[control_cells, target_gene]
            control_pred_target = matrix[control_cells, target_gene]
            recovery_denominator = int(np.sum(condition_truth_target > 0))
            ectopic_denominator = int(np.sum(control_truth_target == 0))
            unit_records.append({
                "unit": target,
                "method": method,
                "target_logfc_abs_error": float(abs(target_predicted - target_reference)),
                "target_direction_preserved": float(np.sign(target_predicted) == np.sign(target_reference)),
                "signature_logfc_spearman": spearman(reference_logfc, predicted_logfc),
                "response_pr_auc": average_precision_tie_aware(response, predicted_logfc),
                "direction_top_decile": float(np.mean(np.sign(reference_logfc[strong]) == np.sign(predicted_logfc[strong]))),
                "target_recovery_rate": float(np.mean(condition_pred_target[condition_truth_target > 0] > 0.5)) if recovery_denominator else float("nan"),
                "ectopic_target_fill_rate": float(np.mean(control_pred_target[control_truth_target == 0] > 0.5)) if ectopic_denominator else float("nan"),
                "n_condition_cells": int(len(condition_cells)),
                "n_control_cells": int(len(control_cells)),
            })

    # Control-vs-control pseudo-contrast: any method that invents strong
    # differential genes in controls has an inflated false-response rate.
    control_test = np.flatnonzero(test & controls)
    if len(control_test) >= 20:
        half = len(control_test) // 2
        first, second = control_test[:half], control_test[half:]
        for method, matrix in matrices.items():
            predicted = log2fc_vector(matrix, matrix[first], matrix[second])
            control_metrics.append({
                "method": method,
                "control_false_response_rate": float(np.mean(np.abs(predicted) > 1.0)),
            })

    unit_metrics = pd.DataFrame(unit_records)
    if unit_metrics.empty:
        raise RuntimeError("no perturbation units survived the minimum cell thresholds")
    metric_columns = [column for column in unit_metrics.columns if column not in {"unit", "method", "n_condition_cells", "n_control_cells"}]
    unit_metrics = unit_metrics.melt(
        id_vars=["unit", "method"],
        value_vars=metric_columns,
        var_name="metric",
        value_name="value",
    )
    units = np.sort(unit_metrics["unit"].unique())
    indices = bootstrap_indices(len(units), args.bootstrap, args.seed)
    summary, comparisons = summarize_units(unit_metrics, indices)
    control_frame = pd.DataFrame(control_metrics)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit_metrics.to_parquet(output / "unit_metrics.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    control_frame.to_parquet(output / "control_metrics.parquet", index=False)
    pd.DataFrame(leakage_checks).to_parquet(output / "leakage_checks.parquet", index=False)
    report = {
        "design": "locked development/test cell split per perturbation; test cells used once",
        "unit": "perturbation condition",
        "n_units": int(len(units)),
        "n_test_cells": int(test.sum()),
        "bootstrap_replicates": args.bootstrap,
        "methods": sorted(unit_metrics["method"].unique()),
        "leakage_checks_passed": True,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
