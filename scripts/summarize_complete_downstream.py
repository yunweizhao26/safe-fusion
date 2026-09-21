#!/usr/bin/env python3
"""Aggregate all five downstream evaluations into range-based conclusions."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("artifacts/paper_evidence/downstream_complete")
OUTPUT = ROOT / "summary"

SOURCES = [
    ("clustering", "pancreas", ROOT / "clustering/pancreas/combined"),
    ("clustering", "colon", ROOT / "clustering/colon"),
    ("markers", "pancreas", ROOT / "pancreas_biology"),
    ("markers", "colon", ROOT / "markers/colon"),
    ("differential_expression", "pancreas", ROOT / "pancreas_biology"),
    ("trajectory", "zebrafish", ROOT / "trajectory/zebrafish"),
    ("grn", "norman_crispra", ROOT / "grn/norman_crispra"),
    ("grn", "adamson_crispri", ROOT / "grn/adamson_crispri"),
    ("grn", "dixit_ko", ROOT / "grn/dixit_ko"),
    ("grn", "papalexi_eccite", ROOT / "grn/papalexi_eccite"),
]

TASK_METRICS = {
    "clustering": {
        "annotation_ari": 1,
        "annotation_nmi": 1,
        "reference_cluster_ari": 1,
        "neighbor_purity": 1,
    },
    "markers": {
        "canonical_marker_pr_auc": 1,
        "canonical_marker_effect_spearman": 1,
        "ectopic_marker_fill_rate": -1,
    },
    "differential_expression": {
        "disease_logfc_spearman": 1,
        "disease_logfc_rmse": -1,
        "disease_direction_top_decile": 1,
    },
    "trajectory": {
        "stage_rank_mae": -1,
        "lineage_accuracy": 1,
        "adjacent_stage_neighbor_rate": 1,
        "stage_pseudobulk_spearman": 1,
        "dynamic_gene_spearman": 1,
    },
    "grn": {
        "edge_pr_auc_q05": 1,
        "edge_pr_auc_q10": 1,
        "edge_roc_auc_q10": 1,
        "effect_spearman_development": 1,
        "effect_spearman_test_truth": 1,
        "effect_rmse_test_truth": -1,
        "direction_accuracy_q10": 1,
        "top10_edge_jaccard": 1,
    },
}


def normalize_method(value: str) -> str:
    return "weighted_knn" if value == "graph_smooth" else value


def budget_from_method(value: str) -> int | None:
    match = re.fullmatch(r"safe_fusion_(\d+)pct", value)
    return int(match.group(1)) if match else None


def relevant_rows(task: str, frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["metric"].isin(TASK_METRICS[task])].copy()
    if task == "markers" and "scope" in frame:
        frame = frame[(frame["scope"].isna()) | (frame["scope"] == "donor")]
    if task == "differential_expression" and "scope" in frame:
        frame = frame[frame["scope"] == "disease_contrast"]
    return frame


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    summaries = []
    comparisons = []
    reports = []
    for task, dataset, folder in SOURCES:
        summary_path = folder / "bootstrap_summary.parquet"
        comparison_path = folder / "paired_comparisons.parquet"
        report_path = folder / "report.json"
        if not summary_path.exists() or not comparison_path.exists() or not report_path.exists():
            raise FileNotFoundError(f"incomplete {task}/{dataset}: {folder}")
        summary = relevant_rows(task, pd.read_parquet(summary_path))
        comparison = relevant_rows(task, pd.read_parquet(comparison_path))
        summary["method"] = summary["method"].map(normalize_method)
        comparison["method"] = comparison["method"].map(normalize_method)
        comparison["reference"] = comparison["reference"].map(normalize_method)
        summary.insert(0, "dataset", dataset)
        summary.insert(0, "task", task)
        comparison.insert(0, "dataset", dataset)
        comparison.insert(0, "task", task)
        summaries.append(summary)
        comparisons.append(comparison)
        reports.append({"task": task, "dataset": dataset, "path": str(report_path), "report": json.loads(report_path.read_text())})
    return pd.concat(summaries, ignore_index=True), pd.concat(comparisons, ignore_index=True), reports


def build_range_conclusions(summary: pd.DataFrame, comparisons: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    grouping = ["task", "dataset", "metric"]
    if "contrast" in summary:
        grouping.append("contrast")
    for keys, block in summary.groupby(grouping, dropna=False, observed=True):
        values = dict(zip(grouping, keys if isinstance(keys, tuple) else (keys,)))
        task = str(values["task"])
        metric = str(values["metric"])
        direction = TASK_METRICS[task][metric]
        method_estimates = block.groupby("method", observed=True)["estimate"].mean()
        if "corrupted_raw" not in method_estimates:
            continue
        raw = float(method_estimates["corrupted_raw"])
        safe = []
        for method, estimate in method_estimates.items():
            budget = budget_from_method(str(method))
            if budget is not None:
                safe.append((budget, str(method), float(estimate)))
        if len(safe) != 10:
            raise ValueError(f"expected ten Safe Fusion fractions for {values}, found {len(safe)}")
        safe.sort()
        best_budget, _, best_estimate = max(safe, key=lambda item: direction * item[2])
        improvements = [direction * (estimate - raw) > 0 for _, _, estimate in safe]
        comparison_block = comparisons[
            (comparisons["task"] == values["task"])
            & (comparisons["dataset"] == values["dataset"])
            & (comparisons["metric"] == values["metric"])
            & (comparisons["reference"] == "corrupted_raw")
        ]
        if "contrast" in grouping:
            target = values["contrast"]
            comparison_block = comparison_block[
                comparison_block["contrast"].fillna("__missing__")
                == ("__missing__" if pd.isna(target) else target)
            ]
        significant = 0
        for _, method, _ in safe:
            match = comparison_block[comparison_block["method"] == method]
            if match.empty:
                continue
            if direction > 0 and float(match.iloc[0]["ci_low"]) > 0:
                significant += 1
            if direction < 0 and float(match.iloc[0]["ci_high"]) < 0:
                significant += 1
        rows.append({
            **values,
            "direction": "higher" if direction > 0 else "lower",
            "corrupted_raw": raw,
            "weighted_knn": float(method_estimates.get("weighted_knn", np.nan)),
            "gene_median": float(method_estimates.get("gene_median", np.nan)),
            "svd": float(method_estimates.get("svd", np.nan)),
            "safe_fusion_min": float(min(value for _, _, value in safe)),
            "safe_fusion_max": float(max(value for _, _, value in safe)),
            "best_safe_fusion_fraction": best_budget / 100.0,
            "best_safe_fusion": best_estimate,
            "fractions_better_than_corrupted": int(sum(improvements)),
            "fractions_significantly_better_than_corrupted": int(significant),
        })
    return pd.DataFrame(rows)


def write_markdown(conclusions: pd.DataFrame, reports: list[dict]) -> None:
    primary = {
        "clustering": "annotation_ari",
        "markers": "canonical_marker_pr_auc",
        "differential_expression": "disease_logfc_spearman",
        "trajectory": "dynamic_gene_spearman",
        "grn": "edge_pr_auc_q10",
    }
    lines = [
        "# Complete downstream analysis",
        "",
        "All five tasks use real experimental data. SERGIO is not used.",
        "Safe Fusion is evaluated at every integer selected fraction from 1 to 10 percent.",
        "",
        "| Task | Dataset or contrast | Raw | Weighted kNN | Safe Fusion range | Fractions better than raw | Significant fractions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task, metric in primary.items():
        subset = conclusions[(conclusions["task"] == task) & (conclusions["metric"] == metric)]
        for row in subset.itertuples():
            contrast = getattr(row, "contrast", np.nan)
            label = row.dataset if pd.isna(contrast) else f"{row.dataset}, {contrast}"
            lines.append(
                f"| {task.replace('_', ' ')} | {label} | {row.corrupted_raw:.4f} | "
                f"{row.weighted_knn:.4f} | {row.safe_fusion_min:.4f} to {row.safe_fusion_max:.4f} | "
                f"{row.fractions_better_than_corrupted}/10 | {row.fractions_significantly_better_than_corrupted}/10 |"
            )
    lines.extend([
        "",
        "The GRN rows measure recovery of intervention-defined response edges. They do not assert direct physical binding.",
        "The perturbation objects lack complete independent replicate fields, so perturbed regulators are the bootstrap units.",
        "",
        "## Verified reports",
        "",
    ])
    lines.extend(f"- {item['task']}, {item['dataset']}: `{item['path']}`" for item in reports)
    (OUTPUT / "ANALYSIS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary, comparisons, reports = load_all()
    conclusions = build_range_conclusions(summary, comparisons)
    summary.to_parquet(OUTPUT / "all_bootstrap_summaries.parquet", index=False)
    summary.to_csv(OUTPUT / "all_bootstrap_summaries.csv", index=False)
    comparisons.to_parquet(OUTPUT / "all_paired_comparisons.parquet", index=False)
    conclusions.to_csv(OUTPUT / "range_conclusions.csv", index=False)
    write_markdown(conclusions, reports)
    report = {
        "tasks_complete": sorted(conclusions["task"].unique()),
        "n_task_dataset_metric_rows": int(len(conclusions)),
        "selected_fractions": [value / 100.0 for value in range(1, 11)],
        "sergio_used": False,
        "reports": reports,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "reports"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
