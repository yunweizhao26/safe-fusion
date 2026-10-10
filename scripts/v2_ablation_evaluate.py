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

from fusion_value_bootstrap import Group, group_statistics, interval, statistics_from_sums
from masked_f1_units import load_unit
from selector_attribution import exact_topk
from v2_ablation_common import (
    ABLATION_ROOT,
    PRODUCTION_BOOSTING,
    PRODUCTION_HIDDEN,
    TEACHERS,
    UNIT_KEYS,
    ablation_variants,
    production_teacher_root,
    units,
)

FILL_FRACTION = 0.05
RANKING_STATISTICS = ("mean_f1_1_to_10", "mean_f1_1_to_2", "average_precision")
VALUE_STATISTICS = ("value_log_error", "error_removed_5pct", "error_removed_5pct_own_selection")

def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores.astype(np.float64), kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank

def log_value(contract: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    value = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols], dtype=np.float64)
    return np.log1p(np.maximum(value, 0.0))

def reproduction(unit, root: Path, data) -> dict:

    variants = {variant.name: variant for variant in ablation_variants(unit, root)}
    production = variants["Production"]
    result = {}
    selector = variants[f"Selector {'-'.join(map(str, PRODUCTION_HIDDEN))}"]
    if selector.scores.exists():
        result["selector_max_abs_score_difference"] = float(np.max(np.abs(np.load(selector.scores) - np.load(production.scores))))
    value = variants[f"Value {PRODUCTION_BOOSTING[0]} leaves, minimum leaf {PRODUCTION_BOOSTING[1]}"]
    if (value.value / "mean.npy").exists():
        result["value_max_abs_difference"] = float(np.max(np.abs(
            np.load(value.value / "mean.npy", mmap_mode="r") - np.load(production.value / "mean.npy", mmap_mode="r"))))
    test = data.split == "test"
    no_crossfit = root / "no_crossfit" / unit.key / "methods"
    for teacher in TEACHERS[1:]:
        if (no_crossfit / teacher / "mean.npy").exists():
            new = np.load(no_crossfit / teacher / "mean.npy", mmap_mode="r")[test]
            old = np.load(production_teacher_root(unit) / teacher / "mean.npy", mmap_mode="r")[test]
            metadata = json.loads((no_crossfit / teacher / "metadata.json").read_text())["parameters"]
            result[f"no_crossfit_{teacher}"] = {
                "test_rows_equal_production": bool(np.array_equal(new, old)),
                **{key: value for key, value in metadata["test_cell_agreement"].items() if key != "production_contract"},
            }
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ablation-root", type=Path, default=ABLATION_ROOT)
    parser.add_argument("--families", nargs="+", default=None, help="Variant families to evaluate (default: all).")
    parser.add_argument("--output-dir", type=Path, default=ABLATION_ROOT / "evaluation")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    all_units = units()
    by_dataset: dict[str, list[str]] = {}
    for key in UNIT_KEYS:
        by_dataset.setdefault(all_units[key].dataset, []).append(key)

    absolute_rows, difference_rows, reproduction_checks = [], [], {}
    for dataset, keys in by_dataset.items():
        loaded = []
        for key in keys:
            unit = all_units[key]
            data = load_unit(unit)
            reproduction_checks[key] = reproduction(unit, args.ablation_root, data)
            rows, cols = np.where((data.counts == 0) & (data.split == "test")[:, None])
            labels = np.char.add(f"{key}:", data.unit_labels[rows].astype(str))
            coordinates = pd.read_parquet(unit.coordinates)
            original = np.zeros(data.counts.shape, dtype=np.float32)
            original[coordinates["cell_index"].to_numpy(int), coordinates["gene_index"].to_numpy(int)] = coordinates["original_value"].to_numpy(np.float32)
            variants = [variant for variant in ablation_variants(unit, args.ablation_root)
                        if args.families is None or variant.family in args.families or variant.family == "production"]
            loaded.append((unit, data, rows, cols, labels, original, variants))
        dataset_units = np.unique(np.concatenate([labels for _, _, _, _, labels, _, _ in loaded]))
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(dataset_units), size=(args.draws, len(dataset_units)))
        weights = np.zeros((args.draws + 1, len(dataset_units)))
        weights[0] = 1.0
        np.add.at(weights, (np.repeat(np.arange(1, args.draws + 1), len(dataset_units)), draws.ravel()), 1.0)

        names = [variant.name for variant in loaded[0][6]]
        families = {variant.name: variant.family for variant in loaded[0][6]}
        ranking_sums: dict[str, dict] = {}
        value_sums: dict[str, np.ndarray] = {}
        for unit, data, rows, cols, labels, original, variants in loaded:
            codes = np.searchsorted(dataset_units, labels)
            positive = data.masked[rows, cols]
            group = Group(unit=unit, rows=rows, cols=cols, labels=positive, codes=codes, unit_index=np.unique(codes))
            y = np.log1p(original[rows[positive], cols[positive]].astype(np.float64))
            positive_codes = codes[positive]
            k = max(1, int(round(FILL_FRACTION * len(rows))))
            production_selected = None
            for variant in variants:
                scores = np.load(variant.scores)
                if len(scores) != len(rows):
                    raise ValueError(f"{unit.key} {variant.name}: {len(scores)} scores for {len(rows)} candidates")
                result = group_statistics(ranks(scores), group, weights)
                result.pop("unit_counts")
                previous = ranking_sums.get(variant.name)
                ranking_sums[variant.name] = result if previous is None else {name: previous[name] + result[name] for name in previous}

                selected = exact_topk(scores, k)[positive]
                if production_selected is None:
                    production_selected = selected
                error = np.abs(log_value(variant.value, rows[positive], cols[positive]) - y)
                per_unit = np.stack([
                    np.bincount(positive_codes, minlength=len(dataset_units)).astype(np.float64),
                    np.bincount(positive_codes, weights=y, minlength=len(dataset_units)),
                    np.bincount(positive_codes, weights=error, minlength=len(dataset_units)),
                    np.bincount(positive_codes, weights=np.where(production_selected, error, y), minlength=len(dataset_units)),
                    np.bincount(positive_codes, weights=np.where(selected, error, y), minlength=len(dataset_units)),
                ])
                value_sums[variant.name] = value_sums.get(variant.name, 0.0) + per_unit
            print(json.dumps({"dataset": dataset, "unit": unit.key, "variants": len(variants)}), flush=True)

        per_variant = {}
        for name in names:
            statistics = statistics_from_sums(ranking_sums[name])
            positives, truth, error, inserted, inserted_own = value_sums[name] @ weights.T
            statistics["value_log_error"] = error / positives
            statistics["error_removed_5pct"] = (truth - inserted) / truth
            statistics["error_removed_5pct_own_selection"] = (truth - inserted_own) / truth
            per_variant[name] = statistics
            print(json.dumps({"dataset": dataset, "variant": name, **{
                statistic: round(100 * float(values[0]) if statistic != "value_log_error" else float(values[0]), 4)
                for statistic, values in statistics.items()}}), flush=True)

        for name, statistics in per_variant.items():
            for statistic, values in statistics.items():
                scale = 1.0 if statistic == "value_log_error" else 100.0
                low, high = interval(values[1:])
                absolute_rows.append({
                    "dataset": dataset, "variant": name, "family": families[name], "statistic": statistic,
                    "estimate": scale * values[0], "lower": scale * low, "upper": scale * high,
                    "n_units": len(dataset_units),
                })
                if name == "Production":
                    continue
                difference = values - per_variant["Production"][statistic]
                low, high = interval(difference[1:])
                difference_rows.append({
                    "dataset": dataset, "variant": name, "family": families[name], "statistic": statistic,
                    "difference": scale * difference[0], "lower": scale * low, "upper": scale * high,
                    "interval_includes_zero": bool(low <= 0.0 <= high), "n_units": len(dataset_units),
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    absolute = pd.DataFrame(absolute_rows)
    differences = pd.DataFrame(difference_rows)
    absolute.to_csv(args.output_dir / "absolute.csv", index=False, float_format="%.5f")
    differences.to_csv(args.output_dir / "paired_differences.csv", index=False, float_format="%.5f")
    summary = {
        "design": "variant minus production on the production test candidates; paired bootstrap of donors or targets",
        "units": list(UNIT_KEYS),
        "draws": args.draws,
        "seed": args.seed,
        "fill_fraction_for_value": FILL_FRACTION,
        "units_of_percentage_points": "F1, average precision and error removed differences are percentage points; "
                                      "value_log_error is in log1p units",
        "reproduction": reproduction_checks,
        "robust": {
            f"{row.variant} | {row.statistic} | {row.dataset}": bool(row.interval_includes_zero)
            for row in differences.itertuples()
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

if __name__ == "__main__":
    main()
