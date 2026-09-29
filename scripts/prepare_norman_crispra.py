#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu

from setup_norman_crispra_experiment import split_within_conditions


def target_effect(target_column: np.ndarray, control_column: np.ndarray, log2fc_min: float, p_max: float) -> dict:
    log2fc = float(np.log2(target_column.mean() + 1.0) - np.log2(control_column.mean() + 1.0))
    try:
        statistic, p_value = mannwhitneyu(target_column, control_column, alternative="two-sided")
    except ValueError as exc:
        return {"passed": False, "reason": f"test_failed:{exc}"}
    passed = bool(log2fc >= log2fc_min and p_value < p_max)
    return {
        "target_log2fc": log2fc, "p_value": float(p_value), "passed": passed,
        "reason": "ok" if passed else "target_effect_too_weak",
    }


def read_rows(counts_dataset, rows: np.ndarray) -> sparse.csr_matrix:
    blocks: list[sparse.csr_matrix] = []
    block = 1000
    for start in range(0, len(rows), block):
        value = counts_dataset[rows[start:start + block]].astype(np.float32)
        blocks.append(sparse.csr_matrix(value))
        del value
    return sparse.vstack(blocks).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-cells-per-condition", type=int, default=60)
    parser.add_argument("--max-control-cells", type=int, default=1000)
    parser.add_argument("--min-cells-per-condition", type=int, default=60)
    parser.add_argument("--target-log2fc-min", type=float, default=0.5)
    parser.add_argument("--target-p-max", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    with h5py.File(args.input, "r") as handle:
        condition_categories = handle["obs/__categories/condition"][:].astype(str)
        condition_codes = handle["obs/condition"][:]
        control_flag = handle["obs/control"][:]
        gene_categories = handle["var/__categories/gene_name"][:].astype(str)
        gene_codes = handle["var/gene_name"][:]
        gene_names = gene_categories[gene_codes]
        counts_dataset = handle["layers/counts"]

        conditions = condition_categories[condition_codes]
        gene_lookup = {gene: index for index, gene in enumerate(gene_names)}
        singles = [value for value in sorted(set(conditions)) if re.fullmatch(r"[^+]+\+ctrl", value)]
        control_positions = np.flatnonzero(control_flag == 1)

        qc_rows: list[dict] = []
        eligible: list[dict] = []
        for condition in singles:
            target = condition[:-5]
            if target not in gene_lookup:
                continue
            row = {"condition": condition, "target": target, "n_cells": int(np.sum(conditions == condition))}
            if row["n_cells"] < args.min_cells_per_condition:
                qc_rows.append({**row, "passed": False, "reason": "too_few_cells"})
            else:
                eligible.append(row)
        sampled: list[int] = []
        for row in sorted(eligible, key=lambda value: value["target"]):
            positions = np.flatnonzero(conditions == row["condition"])
            if len(positions) > args.max_cells_per_condition:
                positions = rng.choice(positions, size=args.max_cells_per_condition, replace=False)
            sampled.extend(positions.tolist())
        if len(control_positions) > args.max_control_cells:
            control_positions = rng.choice(control_positions, size=args.max_control_cells, replace=False)
        sampled_cells = np.asarray(sorted(sampled + control_positions.tolist()), dtype=int)
        sampled_conditions = np.where(control_flag[sampled_cells] == 1, "ctrl", conditions[sampled_cells]).astype(object)
        sampled_counts = read_rows(counts_dataset, sampled_cells)

    sampled_split = split_within_conditions(sampled_conditions, args.test_fraction, args.seed)
    development = sampled_split == "development"
    development_controls = development & (sampled_conditions == "ctrl")
    for row in eligible:
        gene = gene_lookup[row["target"]]
        development_cells = development & (sampled_conditions == row["condition"])
        qc_rows.append({
            **row,
            "qc_perturbed_cells": int(development_cells.sum()),
            "qc_control_cells": int(development_controls.sum()),
            **target_effect(
                sampled_counts[development_cells][:, gene].toarray().ravel(),
                sampled_counts[development_controls][:, gene].toarray().ravel(),
                args.target_log2fc_min,
                args.target_p_max,
            ),
        })
    qc = pd.DataFrame(qc_rows)
    effective = qc[qc["passed"]].copy()
    if effective.empty:
        raise RuntimeError("no perturbation condition passed target-effect QC")
    keep = np.isin(sampled_conditions, [*effective["condition"], "ctrl"])
    selected_cells = sampled_cells[keep]
    split = sampled_split[keep].astype(str)
    counts = sampled_counts[keep]

    selected_conditions_str = np.asarray(sampled_conditions[keep], dtype=str)
    selected_targets = np.where(np.char.equal(selected_conditions_str, "ctrl"), "none", np.char.replace(selected_conditions_str, "+ctrl", ""))
    selected_obs = pd.DataFrame({
        "condition": selected_conditions_str,
        "target": selected_targets,
        "control": np.char.equal(selected_conditions_str, "ctrl").astype(int),
        "cell_id": np.asarray([f"cell_{index}" for index in range(len(selected_cells))]),
        "preassigned_split": split,
    })
    output = ad.AnnData(X=counts, obs=selected_obs, var=pd.DataFrame({"feature_name": gene_names}))
    output.var_names = gene_names
    output.layers["counts"] = counts
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(destination)
    report = {
        "source": str(args.input),
        "modality": "CRISPRa (Norman et al. 2019, GSE133344), K562 cells",
        "seed": args.seed,
        "single_gene_conditions_total": int(len(singles)),
        "effective_conditions": int(len(effective)),
        "cells_total": int(len(selected_cells)),
        "control_cells": int(np.sum(np.char.equal(selected_conditions_str, "ctrl"))),
        "split_counts": pd.Series(split).value_counts().to_dict(),
        "conditions": effective[["condition", "target", "n_cells", "target_log2fc", "p_value"]].to_dict(orient="records"),
        "qc_all_conditions": qc.to_dict(orient="records"),
        "qc_rule": {
            "qc_cells": "development cells of the benchmark split",
            "test_fraction": args.test_fraction,
            "min_cells_per_condition": args.min_cells_per_condition,
            "target_log2fc_min": args.target_log2fc_min,
            "target_p_max": args.target_p_max,
            "max_cells_per_condition": args.max_cells_per_condition,
            "max_control_cells": args.max_control_cells,
        },
        "note": "units are perturbations because the processed file has no replicate column",
    }
    report_path = destination.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
