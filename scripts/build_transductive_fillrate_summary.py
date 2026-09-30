#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from combine_mlp_baseline_curves import normalized_auc, pool_curves

EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"

TRANSDUCTIVE_REPORTS = {
    "Pancreas": [
        EVIDENCE / "review_round2" / "fusion_value" / "selectors" / f"pancreas_{fold}" / "safe_fusion_transductive" / "report.json"
        for fold in range(3)
    ],
    "Colon": [
        EVIDENCE / "review_round3" / "colon_crossfit" / "fusion_value" / "selectors" / f"colon_{fold}" / "safe_fusion_transductive" / "report.json"
        for fold in range(3)
    ],
    "CRISPRa": [
        EVIDENCE / "review_round2" / "fusion_value" / "selectors" / "norman_crispra" / "safe_fusion_transductive" / "report.json"
    ],
}

INDUCTIVE_REPORTS = {
    "Pancreas": [
        EVIDENCE / "review_round2" / "leakage_free" / "pancreas_crossfit" / f"fold_{fold}" / "selector_mlp_biology_range_fullteachers" / "calibration_report.json"
        for fold in range(3)
    ],
    "Colon": [
        EVIDENCE / "review_round3" / "colon_crossfit" / f"fold_{fold}" / "selector_mlp_biology_range_fullteachers" / "calibration_report.json"
        for fold in range(3)
    ],
    "CRISPRa": [
        EVIDENCE / "review_round2" / "leakage_free" / "selector_mlp_biology_range_fullteachers" / "norman_crispra" / "calibration_report.json"
    ],
}
BASELINE_CSV = {
    "Pancreas": EVIDENCE / "review_round2" / "leakage_free" / "selector_f1_fillrate_baselines_1000_points.csv",
    "Colon": EVIDENCE / "review_round3" / "colon_crossfit" / "selector_f1_fillrate_baselines_1000_points.csv",
    "CRISPRa": EVIDENCE / "review_round2" / "leakage_free" / "selector_f1_fillrate_baselines_1000_points.csv",
}
MAIN_COMPARATORS = ("Weighted kNN", "SVD", "ALRA", "SAVER", "MAGIC", "scVI", "scGPT")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE / "review_round4" / "transductive_main")
    args = parser.parse_args()

    frames = []
    for dataset in TRANSDUCTIVE_REPORTS:
        transductive = pool_curves(TRANSDUCTIVE_REPORTS[dataset])
        transductive.insert(0, "comparison", "main")
        transductive.insert(0, "method", "Safe Fusion (transductive)")
        transductive.insert(0, "dataset", dataset)
        frames.append(transductive)

        inductive = pool_curves(INDUCTIVE_REPORTS[dataset])
        inductive.insert(0, "comparison", "main")
        inductive.insert(0, "method", "Safe Fusion (inductive)")
        inductive.insert(0, "dataset", dataset)
        frames.append(inductive)

        baseline = pd.read_csv(BASELINE_CSV[dataset])
        baseline = baseline.loc[
            (baseline["dataset"] == dataset) & baseline["method"].isin(MAIN_COMPARATORS)
        ].copy()
        frames.append(baseline)

    combined = pd.concat(frames, ignore_index=True)
    canonical_coverage = np.linspace(0.001, 1.0, 1000)
    normalized = []
    for (dataset, method), frame in combined.groupby(["dataset", "method"], sort=False):
        frame = frame.sort_values("requested_fill_fraction").reset_index(drop=True)
        if len(frame) != len(canonical_coverage):
            raise ValueError(f"{dataset} {method}: {len(frame)} points, expected 1000")
        observed = frame["requested_fill_fraction"].to_numpy()
        if not np.allclose(observed, canonical_coverage, rtol=0.0, atol=1e-12):
            raise ValueError(f"{dataset} {method} uses an unexpected coverage grid")
        frame["coverage_index"] = np.arange(len(frame))
        frame["requested_fill_fraction"] = canonical_coverage
        normalized.append(frame)
    combined = pd.concat(normalized, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "selector_f1_fillrate_transductive_1000_points.csv", index=False)

    ranges = ((0.001, 0.01), (0.001, 0.02), (0.001, 0.05), (0.001, 0.10), (0.001, 0.20), (0.001, 1.0))
    summary: dict[str, dict] = {}
    for dataset, frame in combined.groupby("dataset", sort=False):
        pivot = frame.pivot(index="coverage_index", columns="method", values="masked_f1").sort_index()
        coverage = canonical_coverage
        dataset_summary: dict[str, object] = {}
        for reference in ("Safe Fusion (transductive)", "Safe Fusion (inductive)"):
            best_competitor = pivot[list(MAIN_COMPARATORS)].max(axis=1)
            best_name = pivot[list(MAIN_COMPARATORS)].idxmax(axis=1)
            family_summary: dict[str, object] = {"competitors": list(MAIN_COMPARATORS)}
            for low, high in ranges:
                keep = (coverage >= low - 1e-12) & (coverage <= high + 1e-12)
                x = coverage[keep]
                mine = pivot.loc[keep, reference].to_numpy()
                best = best_competitor.loc[keep].to_numpy()
                margin = mine - best
                family_summary[f"{low:.3f}_{high:.3f}"] = {
                    "points": int(keep.sum()),
                    "fraction_best": float(np.mean(margin >= 0.0)),
                    "mean_f1_margin": float(np.mean(margin)),
                    "minimum_f1_margin": float(np.min(margin)),
                    "maximum_f1_margin": float(np.max(margin)),
                    "reference_mean_f1_over_coverage": normalized_auc(x, mine),
                    "best_competitor_mean_f1_over_coverage": normalized_auc(x, best),
                    "best_competitor_counts": best_name.loc[keep].value_counts().to_dict(),
                }
            dataset_summary[reference] = family_summary
        summary[str(dataset)] = dataset_summary

    (args.output_dir / "selector_f1_fillrate_transductive_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for dataset, entry in summary.items():
        transductive_100 = entry["Safe Fusion (transductive)"]["0.001_0.100"]["fraction_best"]
        inductive_100 = entry["Safe Fusion (inductive)"]["0.001_0.100"]["fraction_best"]
        print(f"{dataset}: transductive best at {round(100 * transductive_100)}/100 fractions "
              f"(0.1%-10%), inductive at {round(100 * inductive_100)}/100")

if __name__ == "__main__":
    main()
