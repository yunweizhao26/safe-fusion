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

import fusion_value_bootstrap as fvb
from masked_f1_units import load_unit, load_units_manifest

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit-source", nargs=2, action="append", metavar=("MANIFEST", "FUSION_VALUE_ROOT"),
                        type=Path, required=True)
    parser.add_argument("--unit-keys", nargs="+", required=True)
    parser.add_argument("--v2-scores", nargs=2, action="append", metavar=("NAME", "ROOT"), required=True,
                        help="Method name and root with <unit>/test_scores.npy of a new selector.")
    parser.add_argument("--references", nargs="+", required=True, help="Methods whose paired differences are reported.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    new = {name: Path(root) for name, root in args.v2_scores}
    families = {**fvb.method_families(), **{name: "selector_v2" for name in new}}
    methods = list(families)
    unknown = set(args.references) - set(methods)
    if unknown:
        raise ValueError(f"unknown references: {sorted(unknown)}")
    units, roots = {}, {}
    for manifest, root in args.unit_source:
        for unit in load_units_manifest(manifest):
            units[unit.key] = unit
            roots[unit.key] = argparse.Namespace(selector_root=root / "selectors",
                                                 probability_root=root / "nonzero_probability")
    by_dataset: dict[str, list[str]] = {}
    for key in args.unit_keys:
        by_dataset.setdefault(units[key].dataset, []).append(key)

    def ranking(method: str, group: fvb.Group, data) -> np.ndarray:
        if method not in new:
            return fvb.ranking(method, group, data, roots[group.unit.key])
        scores = np.load(new[method] / group.unit.key / "test_scores.npy")
        if len(scores) != len(group.rows):
            raise ValueError(f"{group.unit.key} {method}: {len(scores)} scores for {len(group.rows)} candidates")
        order = np.argsort(-scores.astype(np.float64), kind="stable")
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        return rank

    absolute_rows, difference_rows = [], []
    for dataset, keys in by_dataset.items():
        loaded = []
        for key in keys:
            unit = units[key]
            data = load_unit(unit)
            rows, cols = np.where((data.counts == 0) & (data.split == "test")[:, None])
            loaded.append((unit, data, rows, cols, np.char.add(f"{key}:", data.unit_labels[rows].astype(str))))
        dataset_units = np.unique(np.concatenate([labels for *_, labels in loaded]))
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(dataset_units), size=(args.draws, len(dataset_units)))
        weights = np.zeros((args.draws + 1, len(dataset_units)))
        weights[0] = 1.0
        np.add.at(weights, (np.repeat(np.arange(1, args.draws + 1), len(dataset_units)), draws.ravel()), 1.0)
        per_method = {}
        for method in methods:
            sums = None
            for unit, data, rows, cols, unit_labels in loaded:
                codes = np.searchsorted(dataset_units, unit_labels)
                group = fvb.Group(unit=unit, rows=rows, cols=cols, labels=data.masked[rows, cols],
                                  codes=codes, unit_index=np.unique(codes))
                result = fvb.group_statistics(ranking(method, group, data), group, weights)
                result.pop("unit_counts")
                sums = result if sums is None else {name: sums[name] + result[name] for name in sums}
            per_method[method] = fvb.statistics_from_sums(sums)
            print(json.dumps({"dataset": dataset, "method": method, **{
                name: round(100 * float(values[0]), 3) for name, values in per_method[method].items()}}), flush=True)
        for method, statistics in per_method.items():
            for name, values in statistics.items():
                low, high = fvb.interval(values[1:])
                absolute_rows.append({"dataset": dataset, "method": method, "family": families[method], "statistic": name,
                                      "estimate_percent": 100 * values[0], "lower_percent": 100 * low,
                                      "upper_percent": 100 * high, "n_units": len(dataset_units)})
        for reference in args.references:
            for method, statistics in per_method.items():
                if method == reference:
                    continue
                for name in fvb.STATISTICS:
                    difference = per_method[reference][name] - statistics[name]
                    low, high = fvb.interval(difference[1:])
                    difference_rows.append({"dataset": dataset, "reference": reference, "comparator": method,
                                            "comparator_family": families[method], "statistic": name,
                                            "estimate_pp": 100 * difference[0], "lower_pp": 100 * low,
                                            "upper_pp": 100 * high, "n_units": len(dataset_units)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(absolute_rows).to_csv(args.output_dir / "absolute.csv", index=False, float_format="%.4f")
    pd.DataFrame(difference_rows).to_csv(args.output_dir / "paired_differences.csv", index=False, float_format="%.4f")
    (args.output_dir / "design.json").write_text(json.dumps({
        "new_selectors": {name: str(root) for name, root in new.items()}, "references": args.references,
        "unit_sources": [[str(m), str(r)] for m, r in args.unit_source], "unit_keys": args.unit_keys,
        "draws": args.draws, "seed": args.seed, "windows": {k: list(v) for k, v in fvb.WINDOWS.items()},
    }, indent=2) + "\n")

if __name__ == "__main__":
    main()
