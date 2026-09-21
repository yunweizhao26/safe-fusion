#!/usr/bin/env python3
"""Paired biological-unit bootstrap comparisons between selector methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_pair(value: str) -> tuple[str, str]:
    candidate, reference = value.split("=", 1)
    if not candidate or not reference:
        raise ValueError("pairs must use candidate=reference")
    return candidate, reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-metrics", required=True)
    parser.add_argument("--unit-column", required=True)
    parser.add_argument("--stratify-column")
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    frame = pd.read_parquet(args.unit_metrics)
    unit_columns = [args.unit_column]
    if args.stratify_column:
        unit_columns.append(args.stratify_column)
    units = frame[unit_columns].drop_duplicates().sort_values(args.unit_column)
    if units[args.unit_column].duplicated().any():
        raise ValueError("a biological unit maps to multiple strata")
    unit_order = units[args.unit_column].astype(str).to_numpy()
    strata = (
        units[args.stratify_column].astype(str).to_numpy()
        if args.stratify_column
        else np.repeat("all", len(units))
    )
    rng = np.random.default_rng(args.seed)
    bootstrap_indices: list[np.ndarray] = []
    for _ in range(args.bootstrap):
        selected: list[int] = []
        for stratum in sorted(np.unique(strata)):
            positions = np.flatnonzero(strata == stratum)
            selected.extend(
                rng.choice(positions, size=len(positions), replace=True).tolist()
            )
        bootstrap_indices.append(np.asarray(selected, dtype=int))

    records: list[dict] = []
    for candidate, reference in map(parse_pair, args.pair):
        for metric in sorted(frame["metric"].unique()):
            metric_frame = frame[frame["metric"] == metric].copy()
            metric_frame[args.unit_column] = metric_frame[args.unit_column].astype(str)
            pivot = metric_frame.pivot(
                index=args.unit_column, columns="method", values="value"
            ).reindex(unit_order)
            if candidate not in pivot or reference not in pivot:
                raise ValueError(
                    f"missing {candidate} or {reference} for metric {metric}"
                )
            difference = (
                pivot[candidate].to_numpy(dtype=float)
                - pivot[reference].to_numpy(dtype=float)
            )
            samples = np.asarray([
                np.nanmean(difference[index]) for index in bootstrap_indices
            ])
            records.append({
                "method": candidate,
                "reference": reference,
                "metric": metric,
                "difference": float(np.nanmean(difference)),
                "ci_low": float(np.nanquantile(samples, 0.025)),
                "ci_high": float(np.nanquantile(samples, 0.975)),
                "n_units": int(np.isfinite(difference).sum()),
                "unit_column": args.unit_column,
                "stratify_column": args.stratify_column,
                "bootstrap_replicates": int(args.bootstrap),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
