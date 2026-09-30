#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_matched_baseline_f1_curves import pool_curves, ranked_curve
from masked_f1_units import CURVE_BUDGETS, unit_counts
from sle_common import CASE, FOLDS, LEVELS, METHODS, REPORTED, bootstrap_draws, config, interval, load_fills, load_unit

DATASET = "SLE PBMC"

def pooled_prf(counts: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    selected, true_positive, positives = (weights @ counts).T
    precision = np.divide(true_positive, selected, out=np.full(len(selected), np.nan), where=selected > 0)
    recall = np.divide(true_positive, positives, out=np.full(len(selected), np.nan), where=positives > 0)
    f1 = np.divide(2 * true_positive, selected + positives, out=np.full(len(selected), np.nan), where=selected + positives > 0)
    return precision, recall, f1

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", default="main")
    parser.add_argument("--output-dir", type=Path, default=CASE / "results" / "masked")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    focus = config()["focus"]
    curves = {method: [] for method in METHODS}
    unit_rows, focus_rows, checks = [], [], []
    donor_condition = {}
    for fold in range(FOLDS):
        unit = load_unit(args.set_name, fold, "masked")
        with np.load(unit.directory / "fills" / "candidates.npz") as data:
            rows, cols = data["rows"], data["cols"]
        labels = unit.masked[rows, cols].astype(np.int8)
        donors = unit.obs["donor"].astype(str).to_numpy()[rows]
        lineage = unit.obs["lineage"].astype(str).to_numpy()[rows]
        donor_condition.update(unit.obs.drop_duplicates("donor").set_index("donor")["condition"].astype(str).to_dict())
        for method in METHODS:
            fills = load_fills(unit, method)
            order = np.argsort(-fills["priority"], kind="stable")
            curves[method].append(ranked_curve(order, labels, CURVE_BUDGETS))
            for level in LEVELS:
                selected = fills["first_global"] <= level
                frame = unit_counts(selected, labels, donors)
                frame.insert(0, "fraction", round(0.01 * level, 2))
                frame.insert(0, "method", method)
                frame.insert(0, "group", f"sle_{args.set_name}_fold_{fold}")
                frame.insert(0, "dataset", DATASET)
                unit_rows.append(frame)
            for gene in focus:
                in_gene = cols == unit.genes.index(gene)
                for mode in ("global", "gene"):
                    for level in REPORTED:
                        selected = (fills[f"first_{mode}"] <= level) & (fills["value"] > 0)
                        table = pd.DataFrame({
                            "donor": donors[in_gene], "lineage": lineage[in_gene],
                            "selected": selected[in_gene], "label": labels[in_gene],
                        })
                        table["true_positive"] = table["selected"] & (table["label"] == 1)
                        grouped = table.groupby(["donor", "lineage"]).agg(
                            n_zeros=("label", "size"), n_selected=("selected", "sum"),
                            n_true_positive=("true_positive", "sum"), n_masked_positives=("label", "sum"),
                        ).reset_index()
                        grouped.insert(0, "fill_pct", level)
                        grouped.insert(0, "budget", mode)
                        grouped.insert(0, "gene", gene)
                        grouped.insert(0, "method", method)
                        grouped.insert(0, "fold", fold)
                        focus_rows.append(grouped)
            if method == "Safe Fusion":
                for level in LEVELS:
                    contract = unit.directory / "selector" / f"safe_fusion_calibrated_mlp_topk_{round(0.01 * level, 2)}".replace(".", "p")
                    filled = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols]) != 0
                    ours = (fills["first_global"] <= level) & (fills["value"] > 0)
                    checks.append({"fold": fold, "fill_pct": int(level), "contract_filled": int(filled.sum()),
                                   "score_filled": int(ours.sum()), "disagreements": int((filled != ours).sum())})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows, summary = [], {"dataset": DATASET, "methods": {}, "safe_fusion_contract_checks": checks}
    for method, method_curves in curves.items():
        pooled = pool_curves(method_curves)
        for point in pooled:
            curve_rows.append({"dataset": DATASET, "method": method, **point})
        summary["methods"][method] = {
            "peak_f1": max(point["masked_f1"] for point in pooled),
            "peak_fill_percent": 100 * max(pooled, key=lambda point: point["masked_f1"])["realized_fill_fraction"],
        }
    pd.DataFrame(curve_rows).to_csv(args.output_dir / "masked_f1_curves.csv", index=False)
    counts = pd.concat(unit_rows, ignore_index=True)
    counts.to_parquet(args.output_dir / "masked_f1_unit_counts.parquet", index=False)
    for method, frame in counts.groupby("method"):
        pooled = frame.groupby("fraction")[["n_selected", "n_true_positive", "n_masked_positives"]].sum()
        f1 = 2 * pooled["n_true_positive"] / (pooled["n_selected"] + pooled["n_masked_positives"])
        summary["methods"][method]["f1_percent_at"] = (100 * f1).round(3).to_dict()
        summary["methods"][method]["mean_f1_percent_1_to_10"] = float(100 * f1.mean())

    focus_counts = pd.concat(focus_rows, ignore_index=True)
    focus_counts.to_parquet(args.output_dir / "focus_gene_counts.parquet", index=False)
    donors = np.asarray(sorted(donor_condition))
    strata = np.asarray([donor_condition[donor] for donor in donors])
    weights = bootstrap_draws(donors, strata, args.draws)
    weights = np.vstack([np.ones(len(donors)), weights])
    metric_rows = []
    lineages = sorted(focus_counts["lineage"].unique())
    for (method, gene, mode, level), frame in focus_counts.groupby(["method", "gene", "budget", "fill_pct"]):
        groups = {name: frame[frame["lineage"] == name] for name in lineages}
        groups["non-Treg"] = frame[frame["lineage"] != "Treg"]
        groups["all"] = frame
        results = {}
        for name, block in groups.items():
            per_donor = block.groupby("donor")[["n_selected", "n_true_positive", "n_masked_positives", "n_zeros"]].sum().reindex(donors, fill_value=0)
            precision, recall, f1 = pooled_prf(per_donor[["n_selected", "n_true_positive", "n_masked_positives"]].to_numpy(float), weights)
            results[name] = f1
            metric_rows.append({
                "method": method, "gene": gene, "budget": mode, "fill_pct": level, "cells": name,
                "n_candidates": int(per_donor["n_zeros"].sum()), "n_masked_positives": int(per_donor["n_masked_positives"].sum()),
                "n_selected": int(per_donor["n_selected"].sum()), "n_true_positive": int(per_donor["n_true_positive"].sum()),
                "precision": precision[0], "precision_ci": interval(precision[1:]),
                "recall": recall[0], "recall_ci": interval(recall[1:]),
                "f1": f1[0], "f1_ci": interval(f1[1:]),
            })
        difference = results["Treg"] - results["non-Treg"]
        metric_rows.append({"method": method, "gene": gene, "budget": mode, "fill_pct": level, "cells": "Treg minus non-Treg F1",
                            "f1": difference[0], "f1_ci": interval(difference[1:])})
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "focus_gene_metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))

if __name__ == "__main__":
    main()
