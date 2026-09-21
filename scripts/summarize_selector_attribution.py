#!/usr/bin/env python3
"""Combine disjoint selector-attribution folds into one unit-level summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))
from selector_attribution import summarize_with_bootstrap  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    frames = []
    ranking = []
    for fold, source_value in enumerate(args.input):
        source = Path(source_value)
        frame = pd.read_parquet(source / "unit_metrics.parquet")
        frame["source_fold"] = fold
        frames.append(frame)
        rank = pd.read_parquet(source / "ranking_metrics.parquet")
        rank["source_fold"] = fold
        ranking.append(rank)
    units = pd.concat(frames, ignore_index=True)
    duplicate_key = ["unit", "variant", "value_source", "budget"]
    if units.duplicated(duplicate_key).any():
        raise ValueError("an attribution unit appears in more than one input fold")
    summary, comparisons = summarize_with_bootstrap(units, args.bootstrap, args.seed)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    units.to_parquet(output / "unit_metrics.parquet", index=False)
    pd.concat(ranking, ignore_index=True).to_parquet(output / "ranking_metrics_by_fold.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    report = {
        "design": "combined disjoint biological-unit folds",
        "inputs": [str(Path(value)) for value in args.input],
        "n_units": int(units["unit"].nunique()),
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        "artifacts": [
            "unit_metrics.parquet", "ranking_metrics_by_fold.parquet",
            "bootstrap_summary.parquet", "paired_comparisons.parquet",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
