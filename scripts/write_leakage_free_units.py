#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from masked_f1_units import COLON_METHODS, EVIDENCE, ROOT, Unit, build_units, unit_manifest_entry


def contracts(methods_root: Path, baselines: Path, name: str) -> dict[str, Path]:
    return {
        "Gene median": methods_root / "gene_median",
        "Weighted kNN": methods_root / "graph_smooth",
        "SVD": methods_root / "svd_impute",
        "MAGIC (inductive)": methods_root / "magic_inductive",
        "scVI (inductive)": methods_root / "scvi_inductive",
        "ALRA": baselines / "alra" / name,
        "SAVER": baselines / "saver" / name,
        "MAGIC": baselines / "magic" / name,
        "scVI": baselines / "scvi" / name,
        "scGPT": baselines / "scgpt_mvc" / name,
    }


def leakage_free_units(root: Path, seed: int) -> list[Unit]:
    baselines = root / "baselines"
    units = []
    for fold in range(3):
        fold_root = root / "pancreas_crossfit" / f"fold_{fold}"
        units.append(
            Unit(
                dataset="Pancreas",
                key=f"pancreas_{fold}",
                corrupted=fold_root / "corrupted.h5ad",
                coordinates=fold_root / "coordinates.parquet",
                splits=fold_root / "splits.parquet",
                truth=fold_root / "prepared.h5ad",
                fit_split="development",
                unit_column="donor",
                tie_seed=seed + 100 * fold,
                selector_dir=fold_root / "selector_mlp_biology_range_fullteachers",
                contracts=contracts(fold_root, baselines, f"pancreas_fold_{fold}"),
            )
        )
    units.extend(unit for unit in build_units(EVIDENCE, COLON_METHODS, seed=seed) if unit.key == "colon")
    norman = root / "norman_crispra"
    units.append(
        Unit(
            dataset="CRISPRa",
            key="norman_crispra",
            corrupted=norman / "corrupted.h5ad",
            coordinates=norman / "coordinates.parquet",
            splits=norman / "splits.parquet",
            truth=norman / "prepared.h5ad",
            fit_split="development",
            unit_column="target",
            tie_seed=seed + 2000,
            selector_dir=root / "selector_mlp_biology_range_fullteachers" / "norman_crispra",
            contracts=contracts(norman / "methods", baselines, "norman"),
        )
    )
    return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EVIDENCE / "review_round2" / "leakage_free")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to <root>/units_manifest.json.")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--keys", nargs="+", default=None, help="Write only these unit keys (default: every unit).")
    parser.add_argument(
        "--contract",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "PATH"),
        help="Add a comparator contract (repository-relative path) to the written units. Repeatable.",
    )
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output or root / "units_manifest.json"
    units = [unit for unit in leakage_free_units(root, args.seed) if args.keys is None or unit.key in args.keys]
    for unit in units:
        unit.contracts.update({name: ROOT / path for name, path in args.contract})
    entries = [unit_manifest_entry(unit) for unit in units]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {len(entries)} units to {output}")


if __name__ == "__main__":
    main()
