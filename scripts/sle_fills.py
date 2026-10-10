#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

import numpy as np

from sle_common import METHODS, NOT_FILLED, SEED, first_fill, fill_path, load_unit, candidates, method_values, tie_broken_order

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--app", choices=["masked", "deploy"], required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    unit = load_unit(args.set_name, args.fold, args.app)
    rows, cols = candidates(unit)
    donors = unit.obs["donor"].astype(str).to_numpy()
    donor_names, donor_codes = np.unique(donors[rows], return_inverse=True)
    output = unit.directory / "fills"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "candidates.npz", rows=rows.astype(np.int32), cols=cols.astype(np.int32))
    tie_seed = args.seed + 100 * args.fold
    report = {"unit": str(unit.directory), "candidates": int(len(rows)), "held_out_donors": donor_names.tolist(), "methods": {}}
    for method in METHODS:
        score, value = method_values(unit, method, rows, cols)
        order = tie_broken_order(score, tie_seed)
        priority = np.empty(len(order), dtype=np.float32)
        priority[order] = 1.0 - np.arange(len(order)) / max(len(order) - 1, 1)
        first_global = first_fill(order, None)
        first_donor = first_fill(order, donor_codes)
        first_gene = first_fill(order, cols)
        np.savez(fill_path(unit, method), rows=rows.astype(np.int32), cols=cols.astype(np.int32), value=value,
                 priority=priority, first_global=first_global, first_donor=first_donor, first_gene=first_gene)
        report["methods"][method] = {
            f"selected_{mode}_{level}pct": int((first <= level).sum())
            for mode, first in (("global", first_global), ("donor", first_donor), ("gene", first_gene)) for level in (1, 5, 10)
        } | {
            "selected_with_zero_value_10pct": int(((first_global <= 10) & (value <= 0)).sum())
        }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
