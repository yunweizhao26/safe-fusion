#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from disease_control_common import (
    DRAWS,
    EVIDENCE,
    FRACTIONS,
    METHODS,
    SEED,
    TISSUES,
    filled_mask,
    filled_matrix,
    interval,
    load_heldout,
    output_dir,
    stratified_draws,
)


def condition_rows(values: dict[str, np.ndarray], donor_condition: np.ndarray, spec: dict, draws: np.ndarray, statistic) -> list[dict]:
    rows = []
    levels = [spec["control"], *spec["cases"]]
    point = {level: statistic({k: v[donor_condition == level] for k, v in values.items()}) for level in levels}
    boot = {level: [] for level in levels}
    for draw in draws:
        drawn_condition = donor_condition[draw]
        for level in levels:
            boot[level].append(statistic({k: v[draw][drawn_condition == level] for k, v in values.items()}))
    for level in levels:
        low, high = interval(np.asarray(boot[level]))
        row = {
            "condition": level,
            "n_donors": int(np.sum(donor_condition == level)),
            "estimate": point[level],
            "ci_low": low,
            "ci_high": high,
        }
        if level != spec["control"]:
            difference = np.asarray(boot[level]) - np.asarray(boot[spec["control"]])
            row["difference_from_control"] = point[level] - point[spec["control"]]
            row["difference_ci_low"], row["difference_ci_high"] = interval(difference)
            row["ratio_to_control"] = point[level] / point[spec["control"]] if point[spec["control"]] else float("nan")
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--unit-counts", default=str(EVIDENCE / "masked_f1_unit_counts.parquet"))
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    spec = TISSUES[args.tissue]
    output = output_dir("condition", args.tissue)

    data = load_heldout(args.tissue)
    cells = data.cells
    counts = data.counts
    donors = sorted(cells["donor"].unique())
    donor_condition = cells.groupby("donor")["condition_label"].first().loc[donors].to_numpy()
    draws = stratified_draws(donor_condition, args.draws, args.seed)
    donor_index = pd.Categorical(cells["donor"], categories=donors).codes

    def per_donor(values: np.ndarray) -> np.ndarray:
        return np.bincount(donor_index, weights=values, minlength=len(donors))

    n_cells = per_donor(np.ones(len(cells)))
    library = counts.sum(axis=1, dtype=np.float64)
    zero_fraction = (counts == 0).mean(axis=1)
    mean_of_donor_means = lambda v: float(np.mean(v["value"]))
    depth_rows = []
    for name, values in (("library_size", library), ("zero_fraction", zero_fraction)):
        donor_mean = per_donor(values) / n_cells
        for row in condition_rows({"value": donor_mean}, donor_condition, spec, draws, mean_of_donor_means):
            depth_rows.append({"statistic": name, "scope": "held-out cells", **row})
    if "sample_type" in data.obs:
        every = data.obs.assign(library=data.recorded.sum(axis=1), zero_fraction=(data.recorded == 0).mean(axis=1))
        units = every.groupby(["donor", "sample_type"])[["library", "zero_fraction"]].mean().reset_index()
        unit_type = units["sample_type"].to_numpy()
        sample_spec = {"control": "Heal", "cases": tuple(sorted(set(unit_type) - {"Heal"}))}
        sample_draws = stratified_draws(unit_type, args.draws, args.seed)
        for name, column in (("library_size", "library"), ("zero_fraction", "zero_fraction")):
            for row in condition_rows({"value": units[column].to_numpy()}, unit_type, sample_spec, sample_draws, mean_of_donor_means):
                depth_rows.append({"statistic": name, "scope": "all cells by sample type", **row})
    pd.DataFrame(depth_rows).to_csv(output / "depth.csv", index=False)

    zeros = per_donor((counts == 0).sum(axis=1))
    pooled_rate = lambda v: float(v["filled"].sum() / v["zeros"].sum())
    fill_rows = []
    for method in METHODS:
        for fraction in FRACTIONS:
            filled = per_donor(filled_mask(data, filled_matrix(data, method, fraction)).sum(axis=1))
            for row in condition_rows({"filled": filled, "zeros": zeros}, donor_condition, spec, draws, pooled_rate):
                fill_rows.append({"method": method, "fraction": fraction, **row})
    pd.DataFrame(fill_rows).to_csv(output / "fill_rate.csv", index=False)

    unit_counts = pd.read_parquet(args.unit_counts)
    unit_counts = unit_counts[unit_counts["dataset"] == spec["masked_dataset"]].copy()
    unit_counts["unit"] = unit_counts["unit"].astype(str)
    if set(unit_counts["unit"]) != set(donors):
        raise ValueError("masked-benchmark units differ from the held-out donors")
    f1 = lambda v: float(2 * v["tp"].sum() / (v["selected"].sum() + v["positives"].sum()))
    recall = lambda v: float(v["tp"].sum() / v["positives"].sum())
    fill = lambda v: float(v["selected"].sum() / v["zeros"].sum())
    masked_rows = []
    for method, frame in unit_counts.groupby("method", sort=False):
        pivots = {
            column: frame.pivot(index="unit", columns="fraction", values=column).loc[donors]
            for column in ("n_selected", "n_true_positive", "n_masked_positives", "n_zeros")
        }
        fractions = sorted(pivots["n_selected"].columns)
        for fraction in [*FRACTIONS, "mean_1_to_10"]:
            if fraction == "mean_1_to_10":
                arrays = {key: pivots[name][fractions].to_numpy(dtype=float) for key, name in (
                    ("selected", "n_selected"), ("tp", "n_true_positive"), ("positives", "n_masked_positives"), ("zeros", "n_zeros"))}
                statistic = {
                    "masked_f1": lambda v: float(np.mean(2 * v["tp"].sum(axis=0) / (v["selected"].sum(axis=0) + v["positives"].sum(axis=0)))),
                    "masked_recall": lambda v: float(np.mean(v["tp"].sum(axis=0) / v["positives"].sum(axis=0))),
                    "fill_rate": lambda v: float(np.mean(v["selected"].sum(axis=0) / v["zeros"].sum(axis=0))),
                }
            else:
                column = min(fractions, key=lambda value: abs(value - fraction))
                arrays = {key: pivots[name][column].to_numpy(dtype=float) for key, name in (
                    ("selected", "n_selected"), ("tp", "n_true_positive"), ("positives", "n_masked_positives"), ("zeros", "n_zeros"))}
                statistic = {"masked_f1": f1, "masked_recall": recall, "fill_rate": fill}
            for name, function in statistic.items():
                for row in condition_rows(arrays, donor_condition, spec, draws, function):
                    masked_rows.append({"method": method, "fraction": fraction, "statistic": name, **row})
    pd.DataFrame(masked_rows).to_csv(output / "masked_f1.csv", index=False)
    print(pd.DataFrame(masked_rows).query("statistic == 'masked_f1' and fraction == 'mean_1_to_10'").to_string(index=False))


if __name__ == "__main__":
    main()
