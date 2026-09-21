#!/usr/bin/env python3
"""Fail unless every required real-data downstream deliverable is complete."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path("artifacts/paper_evidence/downstream_complete")


def safe_fusion_fractions(frame: pd.DataFrame) -> set[int]:
    result = set()
    for method in frame["method"].astype(str).unique():
        match = re.fullmatch(r"safe_fusion_(\d+)pct", method)
        if match:
            result.add(int(match.group(1)))
    return result


def require_report(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text())
    if report.get("leakage_checks_passed") is not True:
        raise ValueError(f"leakage checks not passed: {path}")
    return report


def main() -> None:
    result_folders = {
        "clustering_pancreas": ROOT / "clustering/pancreas/combined",
        "clustering_colon": ROOT / "clustering/colon",
        "markers_pancreas_and_de": ROOT / "pancreas_biology",
        "markers_colon": ROOT / "markers/colon",
        "trajectory": ROOT / "trajectory/zebrafish",
        "grn_norman": ROOT / "grn/norman_crispra",
        "grn_adamson": ROOT / "grn/adamson_crispri",
        "grn_dixit": ROOT / "grn/dixit_ko",
        "grn_papalexi": ROOT / "grn/papalexi_eccite",
    }
    reports: dict[str, dict] = {}
    for name, folder in result_folders.items():
        report_path = folder / "report.json"
        if name == "clustering_pancreas":
            report = json.loads(report_path.read_text())
            if not all(item.get("leakage_checks_passed") is True for item in report["fold_reports"]):
                raise ValueError("pancreas clustering fold leakage checks failed")
        else:
            report = require_report(report_path)
        summary_path = folder / "bootstrap_summary.parquet"
        comparison_path = folder / "paired_comparisons.parquet"
        if not summary_path.exists() or not comparison_path.exists():
            raise FileNotFoundError(f"missing bootstrap artifacts in {folder}")
        summary = pd.read_parquet(summary_path)
        if safe_fusion_fractions(summary) != set(range(1, 11)):
            raise ValueError(f"incomplete 1 to 10 percent range in {folder}")
        reports[name] = report

    if reports["clustering_pancreas"].get("n_test_units") != 24:
        raise ValueError("pancreas clustering does not cover all 24 donors")
    if reports["clustering_colon"].get("n_test_units") != 9:
        raise ValueError("colon clustering does not cover all 9 locked donors")
    trajectory = reports["trajectory"]
    if trajectory.get("n_stages") != 12 or trajectory.get("n_test_cells") != 729:
        raise ValueError("trajectory stage or test-cell coverage differs from the protocol")

    expected_grn = {
        "grn_norman": ("gain_of_function", 64),
        "grn_adamson": ("loss_of_function", 30),
        "grn_dixit": ("loss_of_function", 10),
        "grn_papalexi": ("loss_of_function", 24),
    }
    for name, (intervention, regulators) in expected_grn.items():
        report = reports[name]
        if report.get("intervention") != intervention or report.get("n_regulators") != regulators:
            raise ValueError(f"real GRN coverage mismatch for {name}")
        if "sergio" in json.dumps(report).lower():
            raise ValueError(f"SERGIO appears in real GRN report {name}")

    summary_report = json.loads((ROOT / "summary/report.json").read_text())
    expected_tasks = {"clustering", "markers", "differential_expression", "trajectory", "grn"}
    if set(summary_report.get("tasks_complete", [])) != expected_tasks:
        raise ValueError("summary does not contain all five downstream tasks")
    if summary_report.get("sergio_used") is not False:
        raise ValueError("summary does not explicitly exclude SERGIO")
    figure = ROOT / "summary/complete_downstream_5panel.png"
    if not figure.exists() or figure.stat().st_size < 10_000:
        raise ValueError("complete downstream figure is missing or empty")
    print(json.dumps({
        "verified": True,
        "tasks": sorted(expected_tasks),
        "real_grn_regulators": sum(value[1] for value in expected_grn.values()),
        "selected_fractions": list(range(1, 11)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
