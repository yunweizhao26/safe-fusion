#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu


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
        n_cells, n_genes = counts_dataset.shape

        conditions = condition_categories[condition_codes]
        gene_lookup = {gene: index for index, gene in enumerate(gene_names)}
        singles = [value for value in sorted(set(conditions)) if re.fullmatch(r"[^+]+\+ctrl", value)]
        control_positions = np.flatnonzero(control_flag == 1)

        qc_rows: list[dict] = []
        target_gene_set = {gene_lookup[condition[:-5]] for condition in singles if condition[:-5] in gene_lookup}
        control_values: dict[int, list[float]] = {gene: [] for gene in target_gene_set}
        block = 1000
        for start in range(0, len(control_positions), block):
            chunk = counts_dataset[control_positions[start:start + block]]
            for gene in target_gene_set:
                control_values[gene].extend(chunk[:, gene].tolist())
            del chunk
        for condition in singles:
            target = condition[:-5]
            if target not in gene_lookup:
                continue
            gene = gene_lookup[target]
            positions = np.flatnonzero(conditions == condition)
            if len(positions) < args.min_cells_per_condition:
                qc_rows.append({"condition": condition, "target": target, "n_cells": int(len(positions)), "passed": False, "reason": "too_few_cells"})
                continue
            target_column = counts_dataset[positions][:, gene]
            control_column = np.asarray(control_values[gene], dtype=np.float32)
            log2fc = float(np.log2(target_column.mean() + 1.0) - np.log2(control_column.mean() + 1.0))
            try:
                statistic, p_value = mannwhitneyu(target_column, control_column, alternative="two-sided")
                passed = bool(log2fc >= args.target_log2fc_min and p_value < args.target_p_max)
                qc_rows.append({
                    "condition": condition, "target": target, "n_cells": int(len(positions)),
                    "target_log2fc": log2fc, "p_value": float(p_value), "passed": passed,
                    "reason": "ok" if passed else "target_effect_too_weak",
                })
            except ValueError as exc:
                qc_rows.append({"condition": condition, "target": target, "n_cells": int(len(positions)), "passed": False, "reason": f"test_failed:{exc}"})

        qc = pd.DataFrame(qc_rows)
        effective = qc[qc["passed"]].copy()
        if effective.empty:
            raise RuntimeError("no perturbation condition passed target-effect QC")

        selected_cells: list[int] = []
        condition_by_cell: list[str] = []
        for _, row in effective.sort_values("target").iterrows():
            positions = np.flatnonzero(conditions == row["condition"])
            if len(positions) > args.max_cells_per_condition:
                positions = rng.choice(positions, size=args.max_cells_per_condition, replace=False)
            selected_cells.extend(positions.tolist())
            condition_by_cell.extend([row["condition"]] * len(positions))
        if len(control_positions) > args.max_control_cells:
            control_positions = rng.choice(control_positions, size=args.max_control_cells, replace=False)
        selected_cells = np.asarray(sorted(selected_cells + control_positions.tolist()), dtype=int)
        selected_conditions = np.asarray(condition_by_cell + ["ctrl"] * len(control_positions), dtype=object)

        blocks: list[sparse.csr_matrix] = []
        block = 1000
        for start in range(0, len(selected_cells), block):
            value = counts_dataset[selected_cells[start:start + block]].astype(np.float32)
            blocks.append(sparse.csr_matrix(value))
            del value
        counts = sparse.vstack(blocks).tocsr()

    selected_conditions_str = np.asarray(selected_conditions, dtype=str)
    selected_targets = np.where(np.char.equal(selected_conditions_str, "ctrl"), "none", np.char.replace(selected_conditions_str, "+ctrl", ""))
    selected_obs = pd.DataFrame({
        "condition": selected_conditions,
        "target": selected_targets,
        "control": np.char.equal(selected_conditions_str, "ctrl").astype(int),
        "cell_id": np.asarray([f"cell_{index}" for index in range(len(selected_cells))]),
    })
    selected_var = pd.DataFrame({"feature_name": gene_names})

    import anndata as ad
    output = ad.AnnData(X=counts, obs=selected_obs, var=selected_var)
    output.var_names = gene_names
    output.layers["counts"] = counts
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(destination)
    report = {
        "source": str(args.input),
        "modality": "CRISPRa (Norman et al. 2019, GSE133344)",
        "seed": args.seed,
        "single_gene_conditions_total": int(len(singles)),
        "effective_conditions": int(len(effective)),
        "cells_total": int(len(selected_cells)),
        "control_cells": int(np.sum(np.char.equal(selected_conditions_str, "ctrl"))),
        "conditions": effective[["condition", "target", "n_cells", "target_log2fc", "p_value"]].to_dict(orient="records"),
        "qc_rule": {
            "min_cells_per_condition": args.min_cells_per_condition,
            "target_log2fc_min": args.target_log2fc_min,
            "target_p_max": args.target_p_max,
            "max_cells_per_condition": args.max_cells_per_condition,
            "max_control_cells": args.max_control_cells,
        },
        "note": "conditions are A549 single-gene CRISPRa perturbations; units are perturbations because the processed file has no replicate column",
    }
    report_path = destination.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
