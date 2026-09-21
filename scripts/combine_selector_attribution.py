#!/usr/bin/env python3
"""Combine disjoint cross-fit selector-attribution outputs.

The biological units must be held out in exactly one input directory.  Unit
rows are concatenated and bootstrapped once, which preserves the paired
cross-fit comparison and avoids treating folds as independent summary values.
Ranking AUCs remain fold-specific because exact pooled AUCs require raw scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from selector_attribution import summarize_with_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--reference-selector",
        help="Optional paired-comparison reference overriding the input report.",
    )
    args = parser.parse_args()

    reports: list[dict] = []
    unit_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    sources: list[str] = []
    for index, value in enumerate(args.input_dir):
        path = Path(value)
        source = path.name
        if source in sources:
            source = f"{source}_{index}"
        sources.append(source)
        report = json.loads((path / "report.json").read_text())
        reports.append(report)
        units = pd.read_parquet(path / "unit_metrics.parquet")
        units.insert(0, "source", source)
        unit_frames.append(units)
        ranking = pd.read_parquet(path / "ranking_metrics.parquet")
        ranking.insert(0, "source", source)
        ranking_frames.append(ranking)

    contract_fields = (
        "unit_column", "budgets", "variants", "architectures",
        "rank_ensemble", "reference_selector", "feature_names",
    )
    for field in contract_fields:
        values = [report.get(field) for report in reports]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"input reports disagree on {field}: {values}")

    units = pd.concat(unit_frames, ignore_index=True)
    source_count = units[["unit", "source"]].drop_duplicates().groupby("unit")[
        "source"
    ].nunique()
    repeated_units = source_count[source_count > 1]
    if len(repeated_units):
        raise ValueError(
            "cross-fit test units occur in multiple inputs: "
            + ", ".join(map(str, repeated_units.index[:10]))
        )

    reference_selector = (
        args.reference_selector or reports[0]["reference_selector"]
    )
    summary, comparisons = summarize_with_bootstrap(
        units.drop(columns="source"),
        args.bootstrap,
        args.seed,
        reference_selector,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    units.to_parquet(output / "unit_metrics.parquet", index=False)
    pd.concat(ranking_frames, ignore_index=True).to_parquet(
        output / "ranking_metrics_by_fold.parquet", index=False,
    )
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    combined_report = {
        "design": "combined disjoint cross-fit biological units with a shared paired unit bootstrap",
        "input_dirs": [str(Path(value)) for value in args.input_dir],
        "sources": sources,
        "n_units": int(units["unit"].nunique()),
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        **{field: reports[0].get(field) for field in contract_fields},
        "reference_selector": reference_selector,
        "ranking_note": "AUC metrics are retained by fold; exact pooled AUC was not computed without raw scores.",
        "artifacts": [
            "unit_metrics.parquet", "ranking_metrics_by_fold.parquet",
            "bootstrap_summary.parquet", "paired_comparisons.parquet",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(combined_report, indent=2, sort_keys=True) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
