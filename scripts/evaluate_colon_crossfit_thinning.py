#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from evaluate_thinning_transfer import FRACTIONS, analyze, dense, unit_table

LEVELS = ("colon_thinning_050", "colon_thinning_025")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round3/colon_crossfit")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPOSITORY / args.root
    thinning = root / "thinning"

    results, frames, differences = {}, [], []
    for level in LEVELS:
        tables = []
        for fold in range(args.folds):
            key = f"{level}_{fold}"
            dataset = {"comparators": thinning / "thinning_trained/comparators" / key}
            unit = {
                "key": key,
                "splits": root / f"fold_{fold}" / "splits.parquet",
                "thin_root": thinning / "thinning_trained" / key,
                "thin_stacked": thinning / "thinning_trained/stacked" / key,
                "mask_root": thinning / "mask_trained" / key,
            }
            data = ad.read_h5ad(thinning / "data" / key / "corrupted.h5ad")
            truth = ad.read_h5ad(root / f"fold_{fold}" / "prepared.h5ad")
            if not np.array_equal(data.obs_names, truth.obs_names) or not np.array_equal(data.var_names, truth.var_names):
                raise ValueError(f"{key}: thinned and truth orders differ")
            tables.append(unit_table(
                dataset, unit,
                dense(truth.layers["counts"]).astype(np.float32),
                dense(data.layers["corrupted_counts"]).astype(np.float32),
                data.obs_names.astype(str).to_numpy(),
                truth.obs["donor"].astype(str).to_numpy(),
            ))
        results[level] = analyze(tables, args.bootstrap, args.seed)
        frame = pd.DataFrame(results[level]["rows"])
        frame.insert(0, "dataset", level)
        frames.append(frame)
        comparison = pd.DataFrame(results[level]["comparisons"])
        comparison.insert(0, "dataset", level)
        differences.append(comparison)
        print(f"== {level}: {results[level]['composition']}")
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(frame.drop(columns="dataset").round(2).to_string(index=False))
            print(comparison.drop(columns="dataset").round(2).to_string(index=False))
    output = thinning / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output / "transfer_summary.csv", index=False)
    pd.concat(differences, ignore_index=True).to_csv(output / "transfer_paired_differences.csv", index=False)
    (output / "transfer_summary.json").write_text(json.dumps(
        {"bootstrap": args.bootstrap, "seed": args.seed, "fractions": FRACTIONS, "results": results},
        indent=1, default=float) + "\n")

if __name__ == "__main__":
    main()
