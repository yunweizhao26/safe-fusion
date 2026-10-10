#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from compute_matched_baseline_f1_curves import tie_broken_order
from fusion_value_bootstrap import Group, group_statistics, interval, ranking, statistics_from_sums
from fusion_value_selectors import FUSION_VALUE_ROOT, UNIT_KEYS
from masked_f1_units import EVIDENCE, MAIN_COMPARATORS, add_root_arguments, load_unit, units_from_args

EXPECTED_COUNT = "Expected count"
PREVALENCE_MULTIPLES = tuple(round(0.1 * step, 1) for step in range(5, 21))
WINDOW = "mean_f1_prevalence_window"
COMPARISONS = {
    "Safe Fusion": (
        *MAIN_COMPARATORS, "scVI P(X>0)", "SVD (stacked)", "MAGIC (stacked)", "scVI (stacked)", EXPECTED_COUNT,
    ),
    "Safe Fusion (transductive)": ("scVI P(X>0)", "MAGIC (stacked)", "scVI (stacked)"),
}
STATISTICS = ("mean_f1_1_to_10", "mean_f1_1_to_2", "average_precision", WINDOW)
DEFAULT_OUTPUT = EVIDENCE / "review_round3" / "downstream" / "table1_metrics"
PUBLISHED = EVIDENCE / "review_round2" / "leakage_free" / "masked_f1_paired_bootstrap.csv"

def expected_count_rank(group: Group, data) -> np.ndarray:
    training = data.split != "test"
    totals = data.counts[training].sum(axis=0, dtype=np.float64)
    share = totals / totals.sum()
    scores = 1.0 - np.exp(-data.library[group.rows] * share[group.cols])
    order = tie_broken_order(scores, group.unit.tie_seed)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank

def window_statistics(rank: np.ndarray, group: Group, weights: np.ndarray) -> dict[str, np.ndarray]:
    labels = group.labels.astype(bool)
    local = np.searchsorted(group.unit_index, group.codes)
    n_units = len(group.unit_index)
    prevalence = float(labels.mean())
    selected = np.empty((n_units, len(PREVALENCE_MULTIPLES)))
    true_positive = np.empty((n_units, len(PREVALENCE_MULTIPLES)))
    for column, multiple in enumerate(PREVALENCE_MULTIPLES):
        chosen = rank < max(1, int(round(multiple * prevalence * len(rank))))
        selected[:, column] = np.bincount(local[chosen], minlength=n_units)
        true_positive[:, column] = np.bincount(local[chosen & labels], minlength=n_units)
    positives = np.bincount(local[labels], minlength=n_units).astype(np.float64)
    w = weights[:, group.unit_index]
    return {
        "selected": w @ selected,
        "true_positive": w @ true_positive,
        "positives": w @ positives,
        "unit_counts": (selected, true_positive, positives, prevalence),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_root_arguments(parser)
    parser.add_argument("--selector-root", type=Path, default=FUSION_VALUE_ROOT / "selectors")
    parser.add_argument("--probability-root", type=Path, default=FUSION_VALUE_ROOT / "nonzero_probability")
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--unit-keys", nargs="+", default=list(UNIT_KEYS),
        help="Units of the manifest to evaluate; units of one dataset are pooled (default: the Table 1 units).",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    methods = list(dict.fromkeys([*COMPARISONS, *(name for names in COMPARISONS.values() for name in names)]))
    units = {unit.key: unit for unit in units_from_args(args, args.seed)}
    by_dataset: dict[str, list[str]] = {}
    for key in args.unit_keys:
        by_dataset.setdefault(units[key].dataset, []).append(key)

    absolute_rows, difference_rows, fraction_rows, curve_rows, count_frames = [], [], [], [], []
    highest: dict[str, dict] = {}
    for dataset, keys in by_dataset.items():
        loaded = []
        for key in keys:
            unit = units[key]
            data = load_unit(unit)
            rows, cols = np.where((data.counts == 0) & (data.split == "test")[:, None])
            loaded.append((unit, data, rows, cols, np.char.add(f"{key}:", data.unit_labels[rows].astype(str))))
        dataset_units = np.unique(np.concatenate([labels for *_, labels in loaded]))
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(dataset_units), size=(args.draws, len(dataset_units)))
        weights = np.zeros((args.draws + 1, len(dataset_units)))
        weights[0] = 1.0
        np.add.at(weights, (np.repeat(np.arange(1, args.draws + 1), len(dataset_units)), draws.ravel()), 1.0)

        per_method: dict[str, dict[str, np.ndarray]] = {}
        window_f1: dict[str, np.ndarray] = {}
        for method in methods:
            sums, window = None, None
            for unit, data, rows, cols, unit_labels in loaded:
                codes = np.searchsorted(dataset_units, unit_labels)
                group = Group(unit=unit, rows=rows, cols=cols, labels=data.masked[rows, cols],
                              codes=codes, unit_index=np.unique(codes))
                rank = expected_count_rank(group, data) if method == EXPECTED_COUNT else ranking(method, group, data, args)
                result = group_statistics(rank, group, weights)
                result.pop("unit_counts")
                part = window_statistics(rank, group, weights)
                selected, true_positive, positives, prevalence = part.pop("unit_counts")
                if method == methods[0]:
                    fraction_rows.extend({
                        "dataset": dataset, "group": unit.key, "n_candidates": int(len(rows)),
                        "prevalence": prevalence, "multiple": multiple, "fill_fraction": multiple * prevalence,
                        "n_selected": max(1, int(round(multiple * prevalence * len(rows)))),
                    } for multiple in PREVALENCE_MULTIPLES)
                for column, multiple in enumerate(PREVALENCE_MULTIPLES):
                    count_frames.append(pd.DataFrame({
                        "dataset": dataset, "method": method, "group": unit.key,
                        "unit": dataset_units[group.unit_index], "multiple": multiple,
                        "n_selected": selected[:, column].astype(np.int64),
                        "n_true_positive": true_positive[:, column].astype(np.int64),
                        "n_masked_positives": positives.astype(np.int64),
                    }))
                sums = result if sums is None else {name: sums[name] + result[name] for name in sums}
                window = part if window is None else {name: window[name] + part[name] for name in window}
            statistics = statistics_from_sums(sums)
            f1 = 2.0 * window["true_positive"] / (window["selected"] + window["positives"][:, None])
            statistics[WINDOW] = f1.mean(axis=1)
            window_f1[method] = f1[0]
            per_method[method] = statistics
            print(json.dumps({"dataset": dataset, "method": method, **{
                name: round(100 * float(values[0]), 3) for name, values in statistics.items()}}), flush=True)

        for method in methods:
            for column, multiple in enumerate(PREVALENCE_MULTIPLES):
                curve_rows.append({"dataset": dataset, "method": method, "multiple": multiple,
                                   "masked_f1_percent": 100 * window_f1[method][column]})
        best_standard = np.max(np.stack([window_f1[name] for name in MAIN_COMPARATORS]), axis=0)
        highest[dataset] = {
            "window_fractions": len(PREVALENCE_MULTIPLES),
            "safe_fusion_highest_among_standard": int(np.sum(window_f1["Safe Fusion"] > best_standard)),
        }
        for method, statistics in per_method.items():
            for name, values in statistics.items():
                low, high = interval(values[1:])
                absolute_rows.append({
                    "dataset": dataset, "method": method, "statistic": name,
                    "estimate_percent": 100 * values[0], "lower_percent": 100 * low, "upper_percent": 100 * high,
                    "n_units": len(dataset_units),
                })
        for reference, comparators in COMPARISONS.items():
            for method in comparators:
                for name in STATISTICS:
                    difference = per_method[reference][name] - per_method[method][name]
                    low, high = interval(difference[1:])
                    difference_rows.append({
                        "dataset": dataset, "reference": reference, "comparator": method, "statistic": name,
                        "estimate_pp": 100 * difference[0], "lower_pp": 100 * low, "upper_pp": 100 * high,
                        "n_units": len(dataset_units),
                    })

    differences = pd.DataFrame(difference_rows)
    published = pd.read_csv(args.published)
    published = published.loc[published["statistic"] == "mean_difference_1_to_10"]
    check = differences.loc[(differences["reference"] == "Safe Fusion") & (differences["statistic"] == "mean_f1_1_to_10")]
    check = check.merge(published, on=["dataset", "comparator"], suffixes=("", "_published"))
    deviation = float(np.max(np.abs(check[["estimate_pp", "lower_pp", "upper_pp"]].to_numpy()
                                    - check[["estimate_pp_published", "lower_pp_published", "upper_pp_published"]].to_numpy())))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(absolute_rows).to_csv(args.output_dir / "absolute.csv", index=False, float_format="%.4f")
    differences.to_csv(args.output_dir / "paired_differences.csv", index=False, float_format="%.4f")
    pd.DataFrame(fraction_rows).to_csv(args.output_dir / "window_fractions.csv", index=False, float_format="%.6f")
    pd.DataFrame(curve_rows).to_csv(args.output_dir / "window_f1_curves.csv", index=False, float_format="%.4f")
    pd.concat(count_frames, ignore_index=True).to_parquet(args.output_dir / "window_unit_counts.parquet", index=False)
    summary = {
        "draws": args.draws,
        "seed": args.seed,
        "prevalence_multiples": list(PREVALENCE_MULTIPLES),
        "comparisons": {reference: list(names) for reference, names in COMPARISONS.items()},
        "safe_fusion_highest_in_window": highest,
        "table1_reproduction": {
            "rows_compared": int(len(check)),
            "max_abs_deviation_pp": deviation,
            "source": str(args.published.resolve().relative_to(REPOSITORY)),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
