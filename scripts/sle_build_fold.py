#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

def fixed_genes(config: dict) -> list[str]:
    genes = [*config["focus"], *config["treg_suppressive"], *config["sex_x"], *config["sex_y"], *config["interferon"]]
    for members in config["lineage"].values():
        genes.extend(members)
    return list(dict.fromkeys(genes))

def variable_genes(counts: sparse.csr_matrix, fit: np.ndarray, n_genes: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = counts[fit].astype(np.float64)
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(matrix.power(2).mean(axis=0)).ravel() - mean**2, 0)
    detected = np.asarray((matrix > 0).sum(axis=0)).ravel()
    score = np.divide(variance - mean, mean + 1e-8)
    score[detected < max(5, int(0.005 * matrix.shape[0]))] = -np.inf
    ranked = np.argsort(-score, kind="stable")
    return ranked[np.isfinite(score[ranked])][:n_genes], score

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--config", default="configs/sle_treg_genes.json")
    parser.add_argument("--variable-genes", type=int, default=1200)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    prepared = ad.read_h5ad(args.prepared)
    counts = sparse.csr_matrix(prepared.layers["counts"])
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[prepared.obs_names, "split"].to_numpy()
    fit = split == "development"
    config = json.loads(Path(args.config).read_text())
    requested = fixed_genes(config)
    present = [gene for gene in requested if gene in prepared.var_names]
    selected, score = variable_genes(counts, fit, args.variable_genes)
    fixed_index = prepared.var_names.get_indexer(present)
    columns = np.unique(np.concatenate([selected, fixed_index])).astype(int)

    var = prepared.var.iloc[columns].copy()
    var["selection_score"] = score[columns]
    var["variable_gene"] = np.isin(columns, selected)
    var["fixed_gene"] = np.isin(columns, fixed_index)
    fold_counts = counts[:, columns].tocsr()
    truth = ad.AnnData(X=fold_counts.copy(), obs=prepared.obs.copy(), var=var)
    truth.layers["counts"] = fold_counts
    truth.uns["gene_selection"] = {
        "variable_genes": int(len(selected)), "rule": "(variance - mean) / mean on development cells",
        "fixed_genes_present": present, "fixed_genes_missing": sorted(set(requested) - set(present)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    truth.write_h5ad(output)
    detected = np.asarray((fold_counts[fit] > 0).mean(axis=0)).ravel()
    report = {
        "prepared": args.prepared, "splits": args.splits, "output": str(output),
        "n_cells": int(truth.n_obs), "n_genes": int(truth.n_vars), "development_cells": int(fit.sum()),
        "test_cells": int((split == "test").sum()), "variable_genes": int(len(selected)),
        "fixed_genes_present": present, "fixed_genes_missing": sorted(set(requested) - set(present)),
        "fixed_genes_also_variable": sorted(set(present) & set(prepared.var_names[selected])),
        "development_detection_of_fixed_genes": {
            gene: float(detected[list(var.index).index(gene)]) for gene in present
        },
        "zero_fraction": float(1.0 - fold_counts.nnz / np.prod(fold_counts.shape)),
    }
    (output.parent / "genes.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
