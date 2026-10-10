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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file
from safefusion_benchmark.splits import FOLDS, training_folds

def log1p_cp10k(counts: np.ndarray) -> np.ndarray:
    library = counts.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(counts * scale[:, None]).astype(np.float32)

def magic_fit_predict(fit_counts: np.ndarray, new_counts: np.ndarray, seed: int, n_jobs: int,
                      block_cells: int | None = None) -> np.ndarray:

    import magic

    operator = magic.MAGIC(random_state=seed, n_jobs=n_jobs, verbose=0)
    fitted = np.asarray(operator.fit_transform(log1p_cp10k(fit_counts)), dtype=np.float32)
    new = log1p_cp10k(new_counts)
    if block_cells is None:
        transitions = operator.graph.extend_to_data(new)
        imputed = np.asarray(transitions @ fitted, dtype=np.float32)
    else:
        fitted_double = fitted.astype(np.float64)
        imputed = np.empty((len(new), fitted.shape[1]), dtype=np.float32)
        for start in range(0, len(new), block_cells):
            transitions = operator.graph.extend_to_data(new[start:start + block_cells])
            imputed[start:start + block_cells] = transitions.toarray() @ fitted_double
    return np.expm1(np.clip(imputed, 0.0, None))

def scvi_fit_predict(fit_counts: np.ndarray, new_counts: np.ndarray, genes: pd.Index, seed: int, epochs: int,
                     fit_condition: np.ndarray | None = None, new_condition: np.ndarray | None = None) -> np.ndarray:

    import scvi

    scvi.settings.seed = seed
    fit_adata = ad.AnnData(X=sparse.csr_matrix(fit_counts))
    fit_adata.var_names = genes
    batch_key = None
    if fit_condition is not None:
        categories = sorted(set(fit_condition) | set(new_condition))
        fit_adata.obs["condition"] = pd.Categorical(fit_condition, categories=categories)
        batch_key = "condition"
    scvi.model.SCVI.setup_anndata(fit_adata, batch_key=batch_key)
    model = scvi.model.SCVI(fit_adata, n_hidden=128, n_latent=10, n_layers=2, dispersion="gene", gene_likelihood="nb")
    model.train(max_epochs=epochs, accelerator="gpu", devices=1, early_stopping=True, check_val_every_n_epoch=1)
    new_adata = ad.AnnData(X=sparse.csr_matrix(new_counts))
    new_adata.var_names = genes
    if fit_condition is not None:
        new_adata.obs["condition"] = pd.Categorical(new_condition, categories=categories)
    expression = model.get_normalized_expression(new_adata, library_size=1e4, batch_size=256, return_mean=True)
    return np.asarray(expression, dtype=np.float32)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["magic", "scvi"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--condition-column", default=None, help="scVI only: obs column used as a categorical covariate.")
    parser.add_argument("--extension-block-cells", type=int, default=None,
                        help="MAGIC only: extend the fitted graph to new cells this many at a time (bounded memory).")
    args = parser.parse_args()
    if args.condition_column is not None and args.method != "scvi":
        raise ValueError("--condition-column is supported for scVI only")
    if args.extension_block_cells is not None and args.method != "magic":
        raise ValueError("--extension-block-cells is supported for MAGIC only")

    adata = ad.read_h5ad(args.input)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    training = np.isin(split, ["development", "validation"])
    if not training.any() or np.any(split[training] == "test"):
        raise ValueError("invalid training split")
    training_rows = np.flatnonzero(training)
    other_rows = np.flatnonzero(~training)
    genes = adata.var_names.astype(str)
    condition = adata.obs[args.condition_column].astype(str).to_numpy() if args.condition_column else None

    def fit_predict(fit_rows: np.ndarray, new_rows: np.ndarray) -> np.ndarray:
        if args.method == "magic":
            return magic_fit_predict(counts[fit_rows], counts[new_rows], args.seed, args.n_jobs, args.extension_block_cells)
        return scvi_fit_predict(counts[fit_rows], counts[new_rows], genes, args.seed, args.epochs,
                                None if condition is None else condition[fit_rows],
                                None if condition is None else condition[new_rows])

    normalized = np.zeros_like(counts)
    if len(other_rows):
        normalized[other_rows] = fit_predict(training_rows, other_rows)
    folds = training_folds(len(training_rows), args.seed)
    for fold in range(FOLDS):
        normalized[training_rows[folds == fold]] = fit_predict(training_rows[folds != fold], training_rows[folds == fold])
    library = counts.sum(axis=1, dtype=np.float64)
    prediction = (normalized * (library[:, None] / 1e4)).astype(np.float32)

    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = genes.tolist()
    if args.method == "magic":
        import magic
        version = str(magic.__version__)
        settings = {"normalization": "log1p CP10K", "defaults": "magic.MAGIC defaults",
                    "extension": "one kernel transition from each new cell into the fitted graph, then the fitted diffusion"}
        if args.extension_block_cells is not None:
            settings["extension_block_cells"] = args.extension_block_cells
    else:
        import scvi
        version = str(scvi.__version__)
        settings = {"n_hidden": 128, "n_latent": 10, "n_layers": 2, "gene_likelihood": "nb", "maximum_epochs": args.epochs,
                    "extension": "new cells are encoded by the trained model"}
    metadata = {
        "method": args.method,
        "method_version": version,
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
            "fitting_cell_proposals_exclude_own_counts": True,
            "cross_fitting_folds": FOLDS,
            "condition_column": args.condition_column,
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.input),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, prediction, metadata)
    print(json.dumps({"method": args.method, "shape": list(prediction.shape), "parameters": metadata["parameters"]}))

if __name__ == "__main__":
    main()
