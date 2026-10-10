#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from masked_f1_units import (
    CURVE_BUDGETS,
    DATASETS,
    EVIDENCE,
    MAIN_COMPARATORS,
    SUPPLEMENTARY_COMPARATORS,
    TEACHER_COMPARATORS,
    UNIT_FRACTIONS,
    add_root_arguments,
    count_scale_values,
    fraction_name,
    load_unit,
    unit_counts,
    units_from_args,
)


def tie_broken_order(scores: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.lexsort((rng.random(len(scores)), -np.nan_to_num(scores, nan=-np.inf)))


def ranked_curve(order: np.ndarray, labels: np.ndarray, budgets: np.ndarray) -> list[dict[str, float]]:
    cumulative_true = np.cumsum(labels[order], dtype=np.int64)
    positives = int(labels.sum())
    zeros = len(labels)
    curve = []
    for budget in budgets:
        selected = max(1, int(round(float(budget) * zeros)))
        curve.append(
            {
                "requested_fill_fraction": float(budget),
                "n_selected": selected,
                "n_true_positive": int(cumulative_true[selected - 1]),
                "n_masked_positives": positives,
                "n_zeros": zeros,
            }
        )
    return curve


def pool_curves(curves: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    point_count = len(curves[0])
    if any(len(curve) != point_count for curve in curves):
        raise ValueError("Curve lengths differ")
    pooled = []
    for index in range(point_count):
        points = [curve[index] for curve in curves]
        requested = float(points[0]["requested_fill_fraction"])
        if any(abs(float(point["requested_fill_fraction"]) - requested) > 1e-12 for point in points):
            raise ValueError("Curve budgets differ")
        selected = int(sum(point["n_selected"] for point in points))
        true_positive = float(sum(point["n_true_positive"] for point in points))
        positives = int(sum(point["n_masked_positives"] for point in points))
        zeros = int(sum(point["n_zeros"] for point in points))
        precision = true_positive / selected
        recall = true_positive / positives
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        pooled.append(
            {
                "requested_fill_fraction": requested,
                "realized_fill_fraction": selected / zeros,
                "n_selected": selected,
                "n_true_positive": true_positive,
                "n_masked_positives": positives,
                "n_zeros": zeros,
                "masked_precision": precision,
                "masked_recall": recall,
                "masked_f1": f1,
            }
        )
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    add_root_arguments(parser)
    parser.add_argument("--output", type=Path, default=EVIDENCE / "selector_f1_fillrate_baselines_1000_points.csv")
    parser.add_argument("--summary-output", type=Path, default=EVIDENCE / "selector_f1_fillrate_baselines_summary.json")
    parser.add_argument("--unit-output", type=Path, default=EVIDENCE / "masked_f1_unit_counts.parquet")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--comparators",
        nargs="+",
        default=[*MAIN_COMPARATORS, *SUPPLEMENTARY_COMPARATORS, *TEACHER_COMPARATORS],
        help="Comparators to rank (default: the main, supplementary and teacher comparators).",
    )
    args = parser.parse_args()

    comparators = tuple(args.comparators)
    family = {
        **{name: "main" for name in MAIN_COMPARATORS},
        **{name: "supplementary" for name in SUPPLEMENTARY_COMPARATORS},
        **{name: "teacher" for name in TEACHER_COMPARATORS},
    }
    units = units_from_args(args, args.seed)
    datasets = tuple(dataset for dataset in DATASETS if any(unit.dataset == dataset for unit in units))
    unit_curves: dict[str, dict[str, list]] = {dataset: {name: [] for name in comparators} for dataset in datasets}
    unit_rows = []
    scales: dict[str, dict[str, str]] = {dataset: {} for dataset in datasets}
    selection_checks = []
    for unit in units:
        data = load_unit(unit)
        test_cells = np.flatnonzero(data.split == "test")
        local_rows, cols = np.where(data.counts[test_cells] == 0)
        rows = test_cells[local_rows]
        labels = data.masked[rows, cols].astype(np.int8)
        cell_units = data.unit_labels[rows]
        n = len(labels)
        for name in comparators:
            scores, scale = count_scale_values(unit.contracts[name], data, rows, cols)
            scales[unit.dataset][name] = scale
            order = tie_broken_order(scores, unit.tie_seed)
            unit_curves[unit.dataset][name].append(ranked_curve(order, labels, CURVE_BUDGETS))
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(n)
            for fraction in UNIT_FRACTIONS:
                selected = rank < max(1, int(round(fraction * n)))
                frame = unit_counts(selected, labels, cell_units)
                frame.insert(0, "fraction", fraction)
                frame.insert(0, "method", name)
                frame.insert(0, "group", unit.key)
                frame.insert(0, "dataset", unit.dataset)
                unit_rows.append(frame)
        for fraction in UNIT_FRACTIONS:
            contract = unit.selector_dir / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}"
            filled = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols])
            selected = filled != 0
            expected = max(1, int(round(fraction * n)))
            selection_checks.append(
                {"group": unit.key, "fraction": fraction, "selected": int(selected.sum()), "expected": expected}
            )
            frame = unit_counts(selected, labels, cell_units)
            frame.insert(0, "fraction", fraction)
            frame.insert(0, "method", "Safe Fusion")
            frame.insert(0, "group", unit.key)
            frame.insert(0, "dataset", unit.dataset)
            unit_rows.append(frame)

    rows_out = []
    summary: dict[str, dict] = {}
    for dataset in datasets:
        summary[dataset] = {}
        for name in comparators:
            curve = pool_curves(unit_curves[dataset][name])
            comparison = family[name]
            for point in curve:
                rows_out.append({"dataset": dataset, "method": name, "comparison": comparison, **point})
            peak = max(curve, key=lambda point: point["masked_f1"])
            summary[dataset][name] = {
                "comparison": comparison,
                "ranking_scale": "native score" if scales[dataset][name] == "frozen_scgpt_masked_value_score" else "counts",
                "stored_scale": scales[dataset][name],
                "peak_fill_percent": 100.0 * peak["realized_fill_fraction"],
                "peak_f1": peak["masked_f1"],
                "f1_at_2_percent": curve[19]["masked_f1"],
                "f1_at_5_percent": curve[49]["masked_f1"],
                "f1_at_10_percent": curve[99]["masked_f1"],
            }
    mismatched = [check for check in selection_checks if check["selected"] != check["expected"]]
    summary["safe_fusion_selection_checks"] = {"checked": len(selection_checks), "mismatched": mismatched}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(args.output, index=False)
    pd.concat(unit_rows, ignore_index=True).to_parquet(args.unit_output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
