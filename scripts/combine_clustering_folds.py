#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_evaluator():
    path = Path(__file__).with_name("evaluate_unsupervised_clustering.py")
    spec = importlib.util.spec_from_file_location("evaluate_unsupervised_clustering", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    folders = [Path(value) for value in args.fold]
    frames = []
    reports = []
    seen: set[str] = set()
    for fold_index, folder in enumerate(folders):
        frame = pd.read_parquet(folder / "unit_seed_metrics.parquet")
        units = set(frame["unit"].astype(str).unique())
        if seen & units:
            raise ValueError("a biological unit appears in more than one fold")
        seen |= units
        frame["fold"] = fold_index
        frames.append(frame)
        reports.append(json.loads((folder / "report.json").read_text()))
    combined = pd.concat(frames, ignore_index=True)
    evaluator = load_evaluator()
    summary, comparisons = evaluator.bootstrap_summary(combined, args.bootstrap, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output / "unit_seed_metrics.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    report = {
        "design": "donor-disjoint cross-fitted clustering evaluation",
        "n_folds": len(folders),
        "n_test_units": len(seen),
        "bootstrap_replicates": args.bootstrap,
        "fold_reports": reports,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
