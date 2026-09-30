#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file

def proposals(method: str, counts: np.ndarray, training: np.ndarray, genes: pd.Index, args) -> tuple[np.ndarray, dict]:
    rows = np.arange(len(counts))
    if method in {"svd_impute", "graph_smooth"}:
        from run_leakage_safe_method import GraphTeacher, fit_pca, log1p_cpm, svd_reconstruct

        pca = fit_pca(log1p_cpm(counts)[0][training], args.components, args.seed)
        if method == "svd_impute":
            return svd_reconstruct(pca, counts), {"svd_components": int(pca.n_components_)}
        graph = GraphTeacher(counts, training, pca, args.neighbors)

        return graph.smooth(counts, np.full(len(counts), -1)), {"svd_components": int(pca.n_components_),
                                                                 "graph_neighbors": int(args.neighbors)}
    from run_inductive_teacher import magic_fit_predict, scvi_fit_predict

    if method == "magic":
        normalized = magic_fit_predict(counts[training], counts[rows], args.seed, args.n_jobs)
        settings = {"normalization": "log1p CP10K", "defaults": "magic.MAGIC defaults"}
    else:
        normalized = scvi_fit_predict(counts[training], counts[rows], genes, args.seed, args.epochs)
        settings = {"n_hidden": 128, "n_latent": 10, "n_layers": 2, "gene_likelihood": "nb", "maximum_epochs": args.epochs}
    library = counts.sum(axis=1, dtype=np.float64)
    return (normalized * (library[:, None] / 1e4)).astype(np.float32), settings

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["svd_impute", "graph_smooth", "magic", "scvi"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--production-contract", type=Path, required=True,
                        help="The production (cross-fitted) contract of the same teacher on the same input.")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    training = np.isin(split, ["development", "validation"])
    if not training.any() or np.any(split[training] == "test"):
        raise ValueError("invalid training split")
    genes = adata.var_names.astype(str)
    prediction, settings = proposals(args.method, counts, training, genes, args)
    production_metadata = json.loads((args.production_contract / "metadata.json").read_text())
    if production_metadata["cell_ids"] != adata.obs_names.astype(str).tolist():
        raise ValueError("the production contract has another cell order")
    production = np.load(args.production_contract / "mean.npy")
    test = split == "test"
    agreement = {
        "production_contract": str(args.production_contract),
        "refit_test_max_abs_difference": float(np.max(np.abs(prediction[test] - production[test]))),
        "refit_test_log1p_mean_abs_difference": float(np.mean(np.abs(np.log1p(prediction[test]) - np.log1p(production[test])))),
    }
    prediction[test] = production[test]

    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = genes.tolist()
    metadata = {
        "method": args.method,
        "method_version": "v2_ablation_teachers",
        "scale": "counts",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["development", "validation"],
        "training_data": {"cell_ids_sha256": order_hash(np.asarray(cell_ids)[training].tolist())},
        "parameters": {
            **settings,
            "fit_cells": int(training.sum()),
            "heldout_test_cells": int((split == "test").sum()),
            "test_used_for_fit": False,
            "fitting_cell_proposals_exclude_own_counts": False,
            "cross_fitting": "none: one model fitted on all training cells proposes for every training cell, "
                             "so a training cell's proposal uses its own counts",
            "test_cell_proposals": "rows of the production teacher, fitted on the same training cells",
            "test_cell_agreement": agreement,
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.input),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, prediction, metadata)
    print(json.dumps({"method": args.method, "shape": list(prediction.shape), "parameters": metadata["parameters"]}))

if __name__ == "__main__":
    main()
