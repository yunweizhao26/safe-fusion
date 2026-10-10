#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from masked_f1_units import EVIDENCE, SUPPLEMENTARY_COMPARATORS, TEACHER_COMPARATORS, UNIT_FRACTIONS


REPORTED_FRACTIONS = (0.01, 0.02, 0.05, 0.10)


def unit_arrays(frame: pd.DataFrame, units: list[str]) -> dict[str, np.ndarray]:
    pivot = {}
    for column in ("n_selected", "n_true_positive", "n_masked_positives"):
        table = frame.pivot(index="unit", columns="fraction", values=column).reindex(units)
        if table.isna().any().any():
            raise ValueError("unit counts are incomplete")
        pivot[column] = table[list(UNIT_FRACTIONS)].to_numpy(dtype=np.float64)
    return pivot


def pooled_f1(selected: np.ndarray, true_positive: np.ndarray, positives: np.ndarray) -> np.ndarray:
    return 2.0 * true_positive.sum(axis=0) / (selected.sum(axis=0) + positives.sum(axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-counts", type=Path, nargs="+", default=[EVIDENCE / "masked_f1_unit_counts.parquet"])
    parser.add_argument(
        "--stacked-root",
        type=Path,
        default=EVIDENCE / "stacked_selector_baselines",
        help="Directory of stacked_selector_baselines.py outputs; skipped when absent.",
    )
    parser.add_argument("--output", type=Path, default=EVIDENCE / "masked_f1_paired_bootstrap.csv")
    parser.add_argument("--summary-output", type=Path, default=EVIDENCE / "masked_f1_paired_bootstrap.json")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    frames = [pd.read_parquet(path) for path in args.unit_counts]
    if args.stacked_root.exists():
        frames += [pd.read_parquet(path) for path in sorted(args.stacked_root.glob("*/*/unit_counts.parquet"))]
    counts = pd.concat(frames, ignore_index=True)
    counts["unit"] = counts["group"].astype(str) + ":" + counts["unit"].astype(str)
    counts["family"] = np.select(
        [
            counts["method"] == "Safe Fusion",
            counts["method"].str.endswith("(stacked)"),
            counts["method"].isin(SUPPLEMENTARY_COMPARATORS),
            counts["method"].isin(TEACHER_COMPARATORS),
        ],
        ["safe_fusion", "stacked", "supplementary", "teacher"],
        default="untrained",
    )

    rows = []
    summary: dict[str, dict] = {}
    for dataset, frame in counts.groupby("dataset", sort=False):
        units = sorted(frame.loc[frame["method"] == "Safe Fusion", "unit"].unique())
        reference = unit_arrays(frame.loc[frame["method"] == "Safe Fusion"], units)
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(units), size=(args.draws, len(units)))
        reference_f1 = pooled_f1(reference["n_selected"], reference["n_true_positive"], reference["n_masked_positives"])
        summary[dataset] = {"n_units": len(units), "safe_fusion_f1_percent": dict(zip(map(str, UNIT_FRACTIONS), (100 * reference_f1).round(3).tolist()))}
        for method, method_frame in frame.groupby("method", sort=False):
            if method == "Safe Fusion":
                continue
            other = unit_arrays(method_frame, units)
            if not np.array_equal(other["n_masked_positives"], reference["n_masked_positives"]):
                raise ValueError(f"{dataset} {method}: masked positives differ from Safe Fusion")
            point = reference_f1 - pooled_f1(other["n_selected"], other["n_true_positive"], other["n_masked_positives"])
            boot = np.empty((args.draws, len(UNIT_FRACTIONS)))
            for index, draw in enumerate(draws):
                boot[index] = (
                    pooled_f1(reference["n_selected"][draw], reference["n_true_positive"][draw], reference["n_masked_positives"][draw])
                    - pooled_f1(other["n_selected"][draw], other["n_true_positive"][draw], other["n_masked_positives"][draw])
                )
            statistics = {f"difference_at_{fraction:.2f}": UNIT_FRACTIONS.index(fraction) for fraction in REPORTED_FRACTIONS}
            family = method_frame["family"].iloc[0]
            for label, column in statistics.items():
                rows.append({
                    "dataset": dataset, "comparator": method, "family": family, "statistic": label,
                    "estimate_pp": 100 * point[column],
                    "lower_pp": 100 * np.quantile(boot[:, column], 0.025),
                    "upper_pp": 100 * np.quantile(boot[:, column], 0.975),
                    "n_units": len(units),
                })
            mean_boot = boot.mean(axis=1)
            rows.append({
                "dataset": dataset, "comparator": method, "family": family, "statistic": "mean_difference_1_to_10",
                "estimate_pp": 100 * point.mean(),
                "lower_pp": 100 * np.quantile(mean_boot, 0.025),
                "upper_pp": 100 * np.quantile(mean_boot, 0.975),
                "n_units": len(units),
            })
            column = UNIT_FRACTIONS.index(0.05)
            unit_reference = 2 * reference["n_true_positive"][:, column] / (reference["n_selected"][:, column] + reference["n_masked_positives"][:, column])
            unit_other = 2 * other["n_true_positive"][:, column] / (other["n_selected"][:, column] + other["n_masked_positives"][:, column])
            summary[dataset][method] = {
                "family": family,
                "mean_difference_1_to_10_pp": [round(100 * point.mean(), 3), round(100 * np.quantile(mean_boot, 0.025), 3), round(100 * np.quantile(mean_boot, 0.975), 3)],
                "units_with_higher_safe_fusion_f1_at_5pct": int(np.sum(unit_reference > unit_other)),
            }

    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False, float_format="%.4f")
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(table.loc[table["statistic"] == "mean_difference_1_to_10"].to_string(index=False, float_format=lambda value: f"{value:.2f}"))


if __name__ == "__main__":
    main()
