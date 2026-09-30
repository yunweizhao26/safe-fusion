#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sle_common import CASE, FOLDS, METHODS, REPORTED, config, load_fills, load_unit

UNITS = (("main", "deploy"), ("main", "masked"), ("sex", "deploy"), ("cite", "deploy"))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CASE / "results" / "focus_fills")
    args = parser.parse_args()
    focus = config()["focus"]
    rows_out = []
    for set_name, app in UNITS:
        if not (CASE / set_name / "fold_0" / app / "fills" / "Safe_Fusion.npz").exists():
            continue
        for fold in range(FOLDS):
            unit = load_unit(set_name, fold, app)
            lineage = unit.obs["lineage"].astype(str).to_numpy()
            condition = unit.obs["condition"].astype(str).to_numpy()
            library = unit.counts.sum(axis=1, dtype=np.float64)
            development = ~unit.test
            fills = {method: load_fills(unit, method) for method in METHODS}
            for gene in focus:
                g = unit.genes.index(gene)
                share = {}
                for name in np.unique(lineage):
                    members = development & (lineage == name)
                    share[name] = unit.counts[members, g].sum() / max(library[members].sum(), 1.0)
                for method, data in fills.items():
                    in_gene = data["cols"] == g
                    rows = data["rows"][in_gene]
                    expected = library[rows] * np.asarray([share[name] for name in lineage[rows]])
                    value = data["value"][in_gene]
                    labels = unit.masked[rows, g] if unit.masked is not None else np.zeros(len(rows), bool)
                    for cells_name in ("Treg", "other cells"):
                        group = (lineage[rows] == "Treg") if cells_name == "Treg" else (lineage[rows] != "Treg")
                        for cond in sorted(set(condition[rows])):
                            members = group & (condition[rows] == cond)
                            for mode in ("global", "donor", "gene"):
                                for level in REPORTED:
                                    filled = members & (data[f"first_{mode}"][in_gene] <= level) & (value > 0)
                                    rows_out.append({
                                        "set": set_name, "application": app, "fold": fold, "gene": gene, "cells": cells_name,
                                        "condition": cond, "method": method, "budget": mode, "fill_pct": level,
                                        "candidates": int(members.sum()), "filled": int(filled.sum()),
                                        "masked_positives": int(labels[members].sum()),
                                        "filled_masked_positives": int((labels & filled).sum()),
                                        "sum_log1p_inserted": float(np.log1p(value[filled]).sum()),
                                        "sum_log1p_expected": float(np.log1p(expected[filled]).sum()),
                                        "filled_above_one_count": int((value[filled] > 1).sum()),
                                        "median_log1p_inserted": float(np.median(np.log1p(value[filled]))) if filled.any() else np.nan,
                                        "median_log1p_expected": float(np.median(np.log1p(expected[filled]))) if filled.any() else np.nan,
                                    })
            print(f"done {set_name} {app} fold {fold}", flush=True)
    table = pd.DataFrame(rows_out)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "focus_fill_counts_by_fold.csv", index=False)
    keys = ["set", "application", "gene", "cells", "method", "budget", "fill_pct"]
    summed = table.groupby(keys, sort=False)[["candidates", "filled", "masked_positives", "filled_masked_positives",
                                              "sum_log1p_inserted", "sum_log1p_expected", "filled_above_one_count"]].sum().reset_index()
    summed["fill_rate"] = summed["filled"] / summed["candidates"]
    summed["mean_log1p_inserted"] = summed["sum_log1p_inserted"] / summed["filled"].where(summed["filled"] > 0)
    summed["mean_log1p_expected"] = summed["sum_log1p_expected"] / summed["filled"].where(summed["filled"] > 0)
    summed = summed.drop(columns=["sum_log1p_inserted", "sum_log1p_expected"])
    summed.to_csv(args.output_dir / "focus_fill_counts.csv", index=False)
    by_condition = table.groupby(keys[:4] + ["condition"] + keys[4:], sort=False)[["candidates", "filled"]].sum().reset_index()
    by_condition["fill_rate"] = by_condition["filled"] / by_condition["candidates"]
    by_condition.to_csv(args.output_dir / "focus_fill_counts_by_condition.csv", index=False)
    view = summed[(summed["application"] == "deploy")].pivot_table(
        index=["set", "gene", "cells", "budget", "fill_pct"], columns="method", values="filled", sort=False)
    print(view.to_string())

if __name__ == "__main__":
    main()
