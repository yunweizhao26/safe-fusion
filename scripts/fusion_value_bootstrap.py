#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from compute_matched_baseline_f1_curves import tie_broken_order
from fusion_value_selectors import FUSION_VALUE_ROOT, UNIT_KEYS, variants
from masked_f1_units import (
    MAIN_COMPARATORS,
    TEACHER_COMPARATORS,
    UNIT_FRACTIONS,
    add_root_arguments,
    count_scale_values,
    load_unit,
    units_from_args,
)

WINDOWS = {
    "mean_f1_1_to_10": UNIT_FRACTIONS,
    "mean_f1_1_to_2": tuple(round(0.010 + 0.001 * step, 3) for step in range(11)),
}
FRACTIONS = tuple(sorted(set(WINDOWS["mean_f1_1_to_10"]) | set(WINDOWS["mean_f1_1_to_2"])))
STATISTICS = (*WINDOWS, "average_precision")
PROBABILITY_SCORES = {"scVI P(X>0)": "scvi", "SAVER P(X>0)": "saver"}
REFERENCES = ("Safe Fusion", "Safe Fusion (transductive)")
KEY_COMPARATORS = ("scVI (stacked)", "MAGIC (stacked)")
DRAW_CHUNK = 250


@dataclass
class Group:
    unit: object
    rows: np.ndarray
    cols: np.ndarray
    labels: np.ndarray
    codes: np.ndarray
    unit_index: np.ndarray


def method_families() -> dict[str, str]:
    families = {name: "table_1" for name in MAIN_COMPARATORS}
    families.update({name: "inductive_teacher" for name in TEACHER_COMPARATORS})
    families.update({name: "unsupervised_probability" for name in PROBABILITY_SCORES})
    families.update({variant.name: f"selector_{variant.family}" for variant in variants()})
    return families


def ranking(method: str, group: Group, data, args) -> np.ndarray:
    unit = group.unit
    rows, cols = group.rows, group.cols
    selectors = {variant.name: variant for variant in variants()}
    if method in selectors:
        scores = np.load(args.selector_root / unit.key / selectors[method].slug / "test_scores.npy")
        if len(scores) != len(rows):
            raise ValueError(f"{unit.key} {method}: {len(scores)} scores for {len(rows)} candidates")
        order = np.argsort(-scores.astype(np.float64), kind="stable")
    else:
        if method in PROBABILITY_SCORES:
            mean = np.load(args.probability_root / PROBABILITY_SCORES[method] / unit.key / "mean.npy", mmap_mode="r")
            scores = np.asarray(mean[rows, cols], dtype=np.float64)
        else:
            scores, _ = count_scale_values(unit.contracts[method], data, rows, cols)
        order = tie_broken_order(scores, unit.tie_seed)
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def group_statistics(rank: np.ndarray, group: Group, weights: np.ndarray) -> dict[str, np.ndarray]:
    labels = group.labels.astype(bool)
    n_units = len(group.unit_index)
    local = np.searchsorted(group.unit_index, group.codes)
    w = weights[:, group.unit_index]
    n = len(rank)
    selected = np.empty((n_units, len(FRACTIONS)))
    true_positive = np.empty((n_units, len(FRACTIONS)))
    for column, fraction in enumerate(FRACTIONS):
        chosen = rank < max(1, int(round(fraction * n)))
        selected[:, column] = np.bincount(local[chosen], minlength=n_units)
        true_positive[:, column] = np.bincount(local[chosen & labels], minlength=n_units)
    positives = np.bincount(local[labels], minlength=n_units).astype(np.float64)

    positive_rank = rank[labels]
    order = np.argsort(positive_rank)
    positive_rank = positive_rank[order]
    positive_unit = local[labels][order]
    one_hot = np.zeros((len(positive_rank), n_units))
    one_hot[np.arange(len(positive_rank)), positive_unit] = 1.0
    cumulative_true = np.cumsum(one_hot, axis=0)
    cumulative_selected = np.empty_like(cumulative_true)
    for index in range(n_units):
        unit_ranks = np.sort(rank[local == index])
        cumulative_selected[:, index] = np.searchsorted(unit_ranks, positive_rank, side="right")
    ap_numerator = np.empty(len(w))
    for start in range(0, len(w), DRAW_CHUNK):
        chunk = w[start:start + DRAW_CHUNK].T
        true_so_far = cumulative_true @ chunk
        selected_so_far = cumulative_selected @ chunk
        positive_weight = chunk[positive_unit]
        precision = np.divide(true_so_far, selected_so_far, out=np.zeros_like(true_so_far), where=selected_so_far > 0)
        ap_numerator[start:start + DRAW_CHUNK] = (positive_weight * precision).sum(axis=0)
    return {
        "selected": w @ selected,
        "true_positive": w @ true_positive,
        "positives": w @ positives,
        "ap_numerator": ap_numerator,
        "unit_counts": (selected, true_positive, positives),
    }


def statistics_from_sums(sums: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    f1 = 2.0 * sums["true_positive"] / (sums["selected"] + sums["positives"][:, None])
    result = {
        name: f1[:, [FRACTIONS.index(fraction) for fraction in fractions]].mean(axis=1)
        for name, fractions in WINDOWS.items()
    }
    result["average_precision"] = sums["ap_numerator"] / sums["positives"]
    return result


def interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summary_table(absolute: pd.DataFrame, differences: pd.DataFrame) -> pd.DataFrame:
    bounds = {"estimate": "estimate", "lower": "lower", "upper": "upper"}
    parts = [absolute.rename(columns={f"{key}_percent": value for key, value in bounds.items()}).assign(quantity="value (%)")]
    for reference in REFERENCES:
        part = differences.loc[differences["reference"] == reference].rename(columns={
            "comparator": "method", "comparator_family": "family",
            **{f"{key}_pp": value for key, value in bounds.items()},
        })
        parts.append(part.assign(quantity=f"{reference} minus method (pp)"))
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot_table(index=["dataset", "family", "method"], columns=["statistic", "quantity"],
                            values=list(bounds), aggfunc="first")
    wide = wide.reorder_levels([1, 2, 0], axis=1).sort_index(axis=1)
    wide.columns = [f"{statistic} | {quantity} | {bound}" for statistic, quantity, bound in wide.columns]
    return wide.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    add_root_arguments(parser)
    parser.add_argument("--selector-root", type=Path, default=FUSION_VALUE_ROOT / "selectors")
    parser.add_argument("--probability-root", type=Path, default=FUSION_VALUE_ROOT / "nonzero_probability")
    parser.add_argument("--output-dir", type=Path, default=FUSION_VALUE_ROOT / "evaluation")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    families = method_families()
    methods = list(families)
    units = {unit.key: unit for unit in units_from_args(args, args.seed)}
    by_dataset: dict[str, list[str]] = {}
    for key in UNIT_KEYS:
        by_dataset.setdefault(units[key].dataset, []).append(key)

    absolute_rows, difference_rows, count_frames = [], [], []
    key_question: dict[str, dict] = {}
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
        for method in methods:
            sums = None
            for unit, data, rows, cols, unit_labels in loaded:
                codes = np.searchsorted(dataset_units, unit_labels)
                group = Group(unit=unit, rows=rows, cols=cols, labels=data.masked[rows, cols],
                              codes=codes, unit_index=np.unique(codes))
                result = group_statistics(ranking(method, group, data, args), group, weights)
                selected, true_positive, positives = result.pop("unit_counts")
                for column, fraction in enumerate(FRACTIONS):
                    count_frames.append(pd.DataFrame({
                        "dataset": dataset, "method": method, "group": unit.key,
                        "unit": dataset_units[group.unit_index], "fraction": fraction,
                        "n_selected": selected[:, column].astype(np.int64),
                        "n_true_positive": true_positive[:, column].astype(np.int64),
                        "n_masked_positives": positives.astype(np.int64),
                    }))
                sums = result if sums is None else {name: sums[name] + result[name] for name in sums}
            per_method[method] = statistics_from_sums(sums)
            print(json.dumps({"dataset": dataset, "method": method, **{
                name: round(100 * float(values[0]), 3) for name, values in per_method[method].items()}}), flush=True)

        for method, statistics in per_method.items():
            for name, values in statistics.items():
                low, high = interval(values[1:])
                absolute_rows.append({
                    "dataset": dataset, "method": method, "family": families[method], "statistic": name,
                    "estimate_percent": 100 * values[0], "lower_percent": 100 * low, "upper_percent": 100 * high,
                    "n_units": len(dataset_units),
                })
        for reference in REFERENCES:
            for method, statistics in per_method.items():
                if method == reference:
                    continue
                for name in STATISTICS:
                    difference = per_method[reference][name] - statistics[name]
                    low, high = interval(difference[1:])
                    difference_rows.append({
                        "dataset": dataset, "reference": reference, "comparator": method,
                        "comparator_family": families[method], "statistic": name,
                        "estimate_pp": 100 * difference[0], "lower_pp": 100 * low, "upper_pp": 100 * high,
                        "n_units": len(dataset_units),
                    })
                    if method in KEY_COMPARATORS:
                        key_question.setdefault(reference, {}).setdefault(method, {}).setdefault(name, {})[dataset] = {
                            "difference_pp": round(100 * difference[0], 3),
                            "interval_pp": [round(100 * low, 3), round(100 * high, 3)],
                            "interval_above_zero": bool(low > 0),
                        }

    for reference, comparators in key_question.items():
        for comparator, statistics in comparators.items():
            for name, datasets in statistics.items():
                datasets["datasets_with_interval_above_zero"] = sum(
                    value["interval_above_zero"] for value in datasets.values() if isinstance(value, dict)
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    absolute, differences = pd.DataFrame(absolute_rows), pd.DataFrame(difference_rows)
    absolute.to_csv(args.output_dir / "absolute.csv", index=False, float_format="%.4f")
    differences.to_csv(args.output_dir / "paired_differences.csv", index=False, float_format="%.4f")
    summary_table(absolute, differences).to_csv(args.output_dir / "summary_table.csv", index=False, float_format="%.4f")
    pd.concat(count_frames, ignore_index=True).to_parquet(args.output_dir / "unit_counts.parquet", index=False)
    summary = {
        "design": (
            "Every method ranks the candidate zeros of the held-out cells; statistics are recomputed on the same "
            "bootstrap draws of donors (pancreas, colon) or perturbation targets (CRISPRa) for every method."
        ),
        "windows": {name: list(fractions) for name, fractions in WINDOWS.items()},
        "draws": args.draws,
        "seed": args.seed,
        "references": list(REFERENCES),
        "key_question": key_question,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
