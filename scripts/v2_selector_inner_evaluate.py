#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

UNITS = {"Pancreas": ("pancreas_0", "pancreas_1", "pancreas_2"), "Colon": ("colon_0", "colon_1", "colon_2"),
         "CRISPRa": ("norman_crispra",)}
DESIGNS = ("entry", "molecule_0p1", "molecule_0p2", "entry_molecule_0p1", "entry_molecule_0p2")
FEATURES = ("base", "detection")
EVALUATIONS = ("masked", "thinned")
CURRENT = ("entry", "base")
REFERENCES = ("scVI teacher", "scVI teacher P(X>0)")
TIE_MARGIN = 0.05

def changes(design: str, features: str) -> int:
    target = {"entry": 0, "molecule": 1, "entry_molecule": 2}[design.rsplit("_0p", 1)[0]]
    return target + (features == "detection")

def pooled(entries: list[dict]) -> dict:
    tp = np.sum([entry["n_true_positive"] for entry in entries], axis=0)
    selected = np.sum([entry["n_selected"] for entry in entries], axis=0)
    positives = sum(entry["n_positives"] for entry in entries)
    ap = sum(entry["average_precision"] * entry["n_positives"] for entry in entries) / positives
    return {"mean_f1": 100 * float(np.mean(2.0 * tp / (selected + positives))), "average_precision": 100 * ap}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="inner/ directory of the selector v2 study")
    parser.add_argument("--colon-units", nargs="+", default=list(UNITS["Colon"]),
                        help="Units pooled as the colon dataset (default: the donor cross-fit folds).")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    UNITS["Colon"] = tuple(args.colon_units)

    rows = []
    for dataset, units in UNITS.items():
        reports = {(unit, design, features): json.loads((args.root / unit / design / f"selector_{features}" / "report.json").read_text())
                   for unit in units for design in DESIGNS for features in FEATURES}
        for design in DESIGNS:
            for features in FEATURES:
                row = {"dataset": dataset, "variant": f"{design}|{features}", "design": design, "features": features,
                       "changes": changes(design, features)}
                for evaluation in EVALUATIONS:
                    stats = pooled([reports[(unit, design, features)]["evaluation"][evaluation]["selector"] for unit in units])
                    row.update({f"{evaluation}_{name}": value for name, value in stats.items()})
                rows.append(row)
        for reference in REFERENCES:
            row = {"dataset": dataset, "variant": reference, "design": "entry", "features": "untrained", "changes": None}
            for evaluation in EVALUATIONS:
                stats = pooled([reports[(unit, "entry", "base")]["evaluation"][evaluation][reference] for unit in units])
                row.update({f"{evaluation}_{name}": value for name, value in stats.items()})
            rows.append(row)
    table = pd.DataFrame(rows)
    current = table.loc[table["variant"] == "|".join(CURRENT)].set_index("dataset")
    for evaluation in EVALUATIONS:
        for name in ("mean_f1", "average_precision"):
            column = f"{evaluation}_{name}"
            table[f"{column}_minus_current"] = table[column] - table["dataset"].map(current[column])
    table["f1_difference_mean"] = table[["masked_mean_f1_minus_current", "thinned_mean_f1_minus_current"]].mean(axis=1)

    candidates = table.loc[table["features"] != "untrained"]
    scores = candidates.groupby("variant").agg(score=("f1_difference_mean", "mean"), changes=("changes", "first"))
    scores = scores.reset_index().sort_values("score", ascending=False)
    best = float(scores["score"].max())
    eligible = scores.loc[scores["score"] >= best - TIE_MARGIN]
    chosen = eligible.sort_values(["changes", "score"], ascending=[True, False]).iloc[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "ablation.csv", index=False, float_format="%.4f")
    scores.to_csv(args.output_dir / "selection_scores.csv", index=False, float_format="%.4f")
    wide = table.pivot_table(index="variant", columns="dataset", sort=False, values=[
        "masked_mean_f1", "thinned_mean_f1", "masked_mean_f1_minus_current", "thinned_mean_f1_minus_current",
        "masked_average_precision", "thinned_average_precision"])
    wide.columns = [f"{dataset} | {value}" for value, dataset in wide.columns]
    wide = wide.join(scores.set_index("variant"))
    wide.to_csv(args.output_dir / "ablation_wide.csv", float_format="%.4f")
    selection = {
        "rule": ("score = mean over datasets of the mean of the masked and thinned F1 differences (1% to 10%) from "
                 "the current selector; among variants within 0.05 points of the best score, choose the fewest "
                 "changes from the current design, then the highest score"),
        "tie_margin_points": TIE_MARGIN,
        "best_score": best,
        "eligible": eligible.to_dict(orient="records"),
        "chosen": {"variant": chosen["variant"], "score": float(chosen["score"]), "changes": int(chosen["changes"])},
        "scores": scores.to_dict(orient="records"),
    }
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(table.round(3).to_string(index=False))
        print(scores.round(3).to_string(index=False))
    print(json.dumps(selection["chosen"]))

if __name__ == "__main__":
    main()
