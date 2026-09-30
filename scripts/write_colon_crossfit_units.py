#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from masked_f1_units import EVIDENCE, ROOT, Unit, unit_manifest_entry
from write_leakage_free_units import contracts

def colon_crossfit_units(root: Path, folds: int, seed: int) -> list[Unit]:
    units = []
    for fold in range(folds):
        fold_root = root / f"fold_{fold}"
        split = pd.read_parquet(fold_root / "splits.parquet")["split"]
        units.append(
            Unit(
                dataset="Colon",
                key=f"colon_{fold}",
                corrupted=fold_root / "corrupted.h5ad",
                coordinates=fold_root / "coordinates.parquet",
                splits=fold_root / "splits.parquet",
                truth=fold_root / "prepared.h5ad",
                fit_split="validation",
                fit_cells=int(split.isin(["development", "validation"]).sum()),
                unit_column="donor",
                tie_seed=seed + 1000 + 100 * fold,
                selector_dir=fold_root / "selector_mlp_biology_range_fullteachers",
                contracts=contracts(fold_root, root / "baselines", f"colon_fold_{fold}"),
            )
        )
    return units

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EVIDENCE / "review_round3" / "colon_crossfit")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <root>/units_manifest.json.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output or root / "units_manifest.json"
    entries = [unit_manifest_entry(unit) for unit in colon_crossfit_units(root, args.folds, args.seed)]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {len(entries)} units to {output}")

if __name__ == "__main__":
    main()
