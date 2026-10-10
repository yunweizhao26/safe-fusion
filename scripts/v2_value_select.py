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

from v2_value_models import VALUES

VALUE_ROOT = REPOSITORY / "artifacts" / "paper_evidence" / "review_round3" / "value_v2_ablations" / "value"
CURRENT = "conditional"

SELECTION_UNITS = (
    "pancreas_thinning_050_0", "pancreas_thinning_050_1", "pancreas_thinning_050_2", "norman_thinning_050",
    "colon_thinning_050_0", "colon_thinning_050_1", "colon_thinning_050_2",
    "pancreas_0", "pancreas_1", "pancreas_2", "norman_crispra", "colon_0", "colon_1", "colon_2",
)
DATASETS = ("Pancreas", "Colon", "CRISPRa")

STATISTICS = {
    "E_selected": ("thinning", "thinned", True, "count", False),
    "B_selected": ("thinning", "thinned", True, "expected_count", True),
    "M_masked": ("masked", "masked", False, "count", False),
    "E_all_thinned": ("thinning", "thinned", False, "count", False),
    "B_all_thinned": ("thinning", "thinned", False, "expected_count", True),
    "M_thinning_masked": ("thinning", "masked", False, "count", False),
}

def group_sums(frame: pd.DataFrame, reference: str, signed: bool) -> tuple[np.ndarray, pd.DataFrame]:
    target = np.log1p(frame[reference].to_numpy(dtype=np.float64))
    errors = {}
    for value in VALUES:
        difference = np.log1p(np.maximum(frame[f"value:{value}"].to_numpy(dtype=np.float64), 0.0)) - target
        errors[value] = difference if signed else np.abs(difference)
    table = pd.DataFrame(errors).assign(group=frame["group"].to_numpy()).groupby("group", sort=True)
    return table.size().to_numpy(dtype=np.float64), table.sum()

def bootstrap_weights(n_groups: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(draws, n_groups))
    weights = np.zeros((draws + 1, n_groups))
    weights[0] = 1.0
    np.add.at(weights, (np.repeat(np.arange(1, draws + 1), n_groups), picks.ravel()), 1.0)
    return weights

def interval(samples: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inner-root", type=Path, default=VALUE_ROOT / "inner")
    parser.add_argument("--output-dir", type=Path, default=VALUE_ROOT / "selection")
    parser.add_argument("--units", nargs="+", default=list(SELECTION_UNITS), help="Inner units to pool.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    entries = pd.concat([pd.read_parquet(args.inner_root / unit / "entries.parquet") for unit in args.units], ignore_index=True)
    samples: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    rows, paired_rows = [], []
    for dataset in DATASETS:
        for name, (benchmark, entry_type, selected_only, reference, signed) in STATISTICS.items():
            frame = entries[(entries["dataset"] == dataset) & (entries["benchmark"] == benchmark)
                            & (entries["entry_type"] == entry_type)]
            if selected_only:
                frame = frame[frame["selected"]]
            sizes, sums = group_sums(frame, reference, signed)
            weights = bootstrap_weights(len(sizes), args.draws, args.seed)
            denominator = weights @ sizes
            statistic = {value: (weights @ sums[value].to_numpy()) / denominator for value in VALUES}
            samples[(dataset, name)] = statistic
            for value in VALUES:
                low, high = interval(statistic[value][1:])
                rows.append({"dataset": dataset, "statistic": name, "value": value, "estimate": float(statistic[value][0]),
                             "lower": low, "upper": high, "entries": int(sizes.sum()), "groups": int(len(sizes))})
                if value != CURRENT:
                    difference = statistic[value] - statistic[CURRENT]
                    low, high = interval(difference[1:])
                    paired_rows.append({"dataset": dataset, "statistic": name, "value": value, "reference": CURRENT,
                                        "difference": float(difference[0]), "lower": low, "upper": high})

    eligible = {
        value: value == CURRENT or all(
            interval((samples[(d, "E_selected")][value] - samples[(d, "E_selected")][CURRENT])[1:])[0] <= 0.0
            for d in DATASETS)
        for value in VALUES
    }
    mean_abs_bias = {value: float(np.mean([abs(samples[(d, "B_selected")][value][0]) for d in DATASETS])) for value in VALUES}
    mean_masked_error = {value: float(np.mean([samples[(d, "M_masked")][value][0] for d in DATASETS])) for value in VALUES}
    candidates = [value for value in VALUES if eligible[value]]
    best = min(candidates, key=lambda value: mean_abs_bias[value])
    tie_intervals = {}
    for value in candidates:
        if value == best:
            continue
        tie_intervals[value] = {
            d: interval((np.abs(samples[(d, "B_selected")][value]) - np.abs(samples[(d, "B_selected")][best]))[1:])
            for d in DATASETS
        }
        for d, (low, high) in tie_intervals[value].items():
            paired_rows.append({"dataset": d, "statistic": "abs_B_selected", "value": value, "reference": best,
                                "difference": float(abs(samples[(d, "B_selected")][value][0]) - abs(samples[(d, "B_selected")][best][0])),
                                "lower": low, "upper": high})
    tied = [best] + [value for value, bounds in tie_intervals.items() if all(low <= 0.0 <= high for low, high in bounds.values())]
    chosen = min(tied, key=lambda value: mean_masked_error[value])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "inner_statistics.csv", index=False, float_format="%.5f")
    pd.DataFrame(paired_rows).to_csv(args.output_dir / "inner_paired_differences.csv", index=False, float_format="%.5f")
    selection = {
        "rule": [
            "eligible unless the paired 95% interval of E_v - E_conditional lies above zero in some dataset",
            "best: smallest mean |B_v| over datasets among eligible values",
            "tie: eligible values whose paired 95% interval of |B_v| - |B_best| includes zero in every dataset; "
            "choose the smallest mean M_v among tied values",
        ],
        "eligible": eligible,
        "mean_abs_B_selected": mean_abs_bias,
        "mean_M_masked": mean_masked_error,
        "best_by_bias": best,
        "tied_with_best": tied,
        "chosen": chosen,
        "current": CURRENT,
        "draws": args.draws,
        "seed": args.seed,
        "units": sorted(entries["unit"].unique().tolist()),
    }
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    table = pd.DataFrame(rows)
    print(table[table["statistic"].isin(["E_selected", "B_selected", "M_masked"])].to_string(index=False))
    print(json.dumps(selection, indent=2))

if __name__ == "__main__":
    main()
