#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"
METRIC = "edge_pr_auc_q10"
S10_COLUMNS = {
    "Masked": "corrupted_raw", "Unmasked": "reference_truth",
    "Safe Fusion 1%": "safe_fusion_1pct", "Safe Fusion 5%": "safe_fusion_5pct", "Safe Fusion 10%": "safe_fusion_10pct",
    "SVD 5%": "svd_5pct", "SVD 10%": "svd_10pct", "Weighted kNN 5%": "weighted_knn_5pct",
    "Weighted kNN 10%": "weighted_knn_10pct", "Dense SVD": "svd", "Dense kNN": "weighted_knn",
}
TABLE4_COLUMNS = {
    f"{name} {pct}%": f"{source}_{pct}pct"
    for name, source in (("Safe Fusion", "safe_fusion"), ("SVD", "svd"), ("Weighted kNN", "weighted_knn"))
    for pct in (1, 10)
}

def benchmark_rows(label: str, masked_grn: Path, deployment_grn: Path, rule_masked: Path, rule_deployment: Path) -> list[dict]:
    rows = []
    boot = pd.read_parquet(masked_grn / "bootstrap_summary.parquet")
    boot = boot.loc[boot["metric"] == METRIC].set_index("method")
    for column, method in S10_COLUMNS.items():
        value = boot.loc[method]
        rows.append({"benchmark": label, "table": "S10", "column": column, "value": value["estimate"],
                     "ci_low": value["ci_low"], "ci_high": value["ci_high"], "n_units": int(value["n_units"])})
    paired = pd.read_parquet(deployment_grn / "paired_comparisons.parquet")
    paired = paired.loc[(paired["metric"] == METRIC) & (paired["reference"] == "corrupted_raw")].set_index("method")
    for column, method in TABLE4_COLUMNS.items():
        value = paired.loc[method]
        rows.append({"benchmark": label, "table": "4", "column": column, "value": value["difference"],
                     "ci_low": value["ci_low"], "ci_high": value["ci_high"], "n_units": int(value["n_units"]),
                     "interval_excludes_zero": bool(value["ci_low"] > 0 or value["ci_high"] < 0)})
    masked = json.loads((rule_masked / "calibration_report.json").read_text())["detection_rule"]
    deployment = json.loads((rule_deployment / "calibration_report.json").read_text())["detection_rule"]
    for column, value in (
        ("Chosen (%)", 100 * masked["test_fill_fraction"]), ("F1", 100 * masked["test_masked_f1"]),
        ("Best (%)", 100 * masked["best_fill_fraction_in_hindsight"]),
        ("Best F1", 100 * masked["best_masked_f1_in_hindsight"]),
        ("Recorded zeros filled (%)", 100 * deployment["test_fill_fraction"]),
    ):
        rows.append({"benchmark": label, "table": "S11", "column": column, "value": value})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EVIDENCE / "review_round2" / "norman_rebuilt")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <root>/norman_downstream_rows.csv.")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPOSITORY / args.root

    rows = benchmark_rows(
        "production",
        EVIDENCE / "downstream_complete" / "grn" / "norman_crispra",
        EVIDENCE / "downstream_deployment" / "evaluation" / "grn" / "norman_crispra",
        EVIDENCE / "detection_rule" / "norman_crispra",
        EVIDENCE / "downstream_deployment" / "norman_crispra" / "detection_rule",
    ) + benchmark_rows(
        "rebuilt",
        root / "downstream_masked" / "grn" / "norman_crispra",
        root / "deployment" / "evaluation" / "grn" / "norman_crispra",
        root / "detection_rule" / "norman_crispra",
        root / "deployment" / "norman_crispra" / "detection_rule",
    )
    table = pd.DataFrame(rows)
    output = args.output or root / "norman_downstream_rows.csv"
    table.to_csv(output, index=False, float_format="%.4f")
    wide = table.pivot_table(index=["table", "column"], columns="benchmark", values="value", sort=False)
    print(wide.to_string(float_format=lambda value: f"{value:.3f}"))

if __name__ == "__main__":
    main()
