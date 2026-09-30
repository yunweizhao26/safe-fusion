#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
VALUE_ROOT = REPOSITORY / "artifacts" / "paper_evidence" / "review_round3" / "value_v2_ablations" / "value"
CURRENT = "safe_fusion"
PCTS = (1, 5, 10)

ENDPOINTS = (
    ("Colon marker AUPRC", "markers/colon", "canonical_marker_pr_auc", None, 100.0, "donor_metrics.parquet", "donor"),
    ("Colon off-target fill", "markers/colon", "ectopic_marker_fill_rate", None, 100.0, "donor_metrics.parquet", "donor"),
    ("Colon reference mapping F1", "markers/colon", "cell_identity_macro_f1", None, 100.0, "donor_metrics.parquet", "donor"),
    ("Pancreas marker AUPRC", "pancreas_biology", "canonical_marker_pr_auc", None, 100.0, "donor_metrics.parquet", "donor"),
    ("Pancreas off-target fill", "pancreas_biology", "ectopic_marker_fill_rate", None, 100.0, "donor_metrics.parquet", "donor"),
    ("Pancreas reference mapping F1", "pancreas_biology", "cell_identity_macro_f1", None, 100.0, "donor_metrics.parquet", "donor"),
    ("Pancreas disease effects, autoantibody", "pancreas_biology", "disease_logfc_spearman", "AAB_vs_Control", 1.0, None, None),
    ("Pancreas disease effects, type 1 diabetes", "pancreas_biology", "disease_logfc_spearman", "T1D_vs_Control", 1.0, None, None),
    ("Zebrafish dynamics", "trajectory/zebrafish", "dynamic_gene_spearman", None, 1.0, "unit_metrics.parquet", "unit"),
    ("Zebrafish stage error", "trajectory/zebrafish", "stage_rank_mae", None, 1.0, "unit_metrics.parquet", "unit"),
    ("Norman response edge AUPRC", "grn/norman_crispra", "edge_pr_auc_q10", None, 1.0, "unit_metrics.parquet", "unit"),
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--value-root", type=Path, default=VALUE_ROOT)
    parser.add_argument("--values", nargs="+", required=True, help="Inserted values of the new fills, for example rate_poisson.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    evaluation = args.value_root / "deployment" / "evaluation"
    output = args.value_root / "deployment" / "summary"
    output.mkdir(parents=True, exist_ok=True)

    rows, paired_rows = [], []
    for label, subpath, metric, contrast, scale, unit_file, unit_column in ENDPOINTS:
        path = evaluation / subpath / "paired_comparisons.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[(frame["metric"] == metric) & (frame["reference"] == "corrupted_raw")]
        if "contrast" in frame:
            frame = frame[frame["contrast"] == (contrast or "all_conditions")]
        if "scope" in frame and contrast is None:
            frame = frame[frame["scope"] == "donor"]
        names = {CURRENT: "conditional", **{f"safe_fusion_{value}": value for value in args.values}}
        for name, value in names.items():
            for pct in PCTS:
                hit = frame[frame["method"] == f"{name}_{pct}pct"]
                if hit.empty:
                    continue
                record = hit.iloc[0]
                rows.append({"endpoint": label, "value": value, "fill_pct": pct,
                             "difference": record["difference"] * scale, "ci_low": record["ci_low"] * scale,
                             "ci_high": record["ci_high"] * scale,
                             "excludes_zero": bool(record["ci_low"] > 0 or record["ci_high"] < 0)})
        if unit_file is None:
            continue
        units = pd.read_parquet(evaluation / subpath / unit_file)
        units = units[units["metric"] == metric]
        for value in args.values:
            for pct in PCTS:
                new = f"safe_fusion_{value}_{pct}pct"
                pivot = units[units["method"].isin([f"{CURRENT}_{pct}pct", new])].pivot_table(
                    index=unit_column, columns="method", values="value", aggfunc="mean")
                if pivot.shape[1] < 2:
                    continue
                difference = (pivot[new] - pivot[f"{CURRENT}_{pct}pct"]).dropna().to_numpy()
                rng = np.random.default_rng(args.seed)
                draws = rng.integers(0, len(difference), size=(args.draws, len(difference)))
                samples = difference[draws].mean(axis=1)
                paired_rows.append({"endpoint": label, "value": value, "fill_pct": pct, "units": len(difference),
                                    "new_minus_current": float(difference.mean()) * scale,
                                    "ci_low": float(np.quantile(samples, 0.025)) * scale,
                                    "ci_high": float(np.quantile(samples, 0.975)) * scale})
    table = pd.DataFrame(rows)
    paired = pd.DataFrame(paired_rows)
    table.to_csv(output / "table3.csv", index=False, float_format="%.5f")
    paired.to_csv(output / "paired.csv", index=False, float_format="%.5f")
    print(table.to_string(index=False))
    print(paired.to_string(index=False))

if __name__ == "__main__":
    main()
