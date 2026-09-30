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

def scvi_fit_predict(fit_counts: np.ndarray, new_counts: np.ndarray, genes: pd.Index, seed: int, epochs: int,
                     fit_condition: np.ndarray | None = None,
                     new_condition: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
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
    theta = np.exp(model.module.px_r.detach().cpu().numpy().astype(np.float64))
    return np.asarray(expression, dtype=np.float32), theta

def dispersions_of_saved_model(contract: Path, output: Path) -> None:
    import torch

    metadata = json.loads((contract / "metadata.json").read_text())
    state = torch.load(contract / "model" / "model.pt", map_location="cpu", weights_only=False)
    settings = state["attr_dict"]["init_params_"]["non_kwargs"]
    if settings["dispersion"] != "gene" or settings["gene_likelihood"] != "nb":
        raise ValueError(f"expected gene dispersion and a negative binomial, got {settings}")
    if list(state["var_names"]) != metadata["gene_ids"]:
        raise ValueError("gene order of the saved model differs from the contract")
    theta = np.exp(state["model_state_dict"]["px_r"].numpy().astype(np.float64))[None, :]
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "theta.npy", theta, allow_pickle=False)
    np.save(output / "model_index.npy", np.zeros(len(metadata["cell_ids"]), dtype=np.int8), allow_pickle=False)
    (output / "metadata.json").write_text(json.dumps({
        "source_contract": str(contract), "theta": "exp(px_r) of the saved standard scVI model, one row for every cell",
        "cell_ids": metadata["cell_ids"], "gene_ids": metadata["gene_ids"]}, indent=1) + "\n")
    print(json.dumps({"source": str(contract), "theta_median": float(np.median(theta))}))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--coordinates")
    parser.add_argument("--splits")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--condition-column", default=None)
    parser.add_argument("--from-model", type=Path, default=None, help="standard scVI contract with model/model.pt")
    args = parser.parse_args()
    if args.from_model is not None:
        dispersions_of_saved_model(args.from_model, Path(args.output))
        return
    if not (args.input and args.coordinates and args.splits):
        parser.error("--input, --coordinates and --splits are required without --from-model")

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

    def fit_predict(fit_rows: np.ndarray, new_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return scvi_fit_predict(counts[fit_rows], counts[new_rows], genes, args.seed, args.epochs,
                                None if condition is None else condition[fit_rows],
                                None if condition is None else condition[new_rows])

    normalized = np.zeros_like(counts)
    theta = np.full((FOLDS + 1, counts.shape[1]), np.nan)
    model_index = np.zeros(len(counts), dtype=np.int8)
    if len(other_rows):
        normalized[other_rows], theta[0] = fit_predict(training_rows, other_rows)
    folds = training_folds(len(training_rows), args.seed)
    for fold in range(FOLDS):
        held = training_rows[folds == fold]
        normalized[held], theta[1 + fold] = fit_predict(training_rows[folds != fold], held)
        model_index[held] = 1 + fold
    library = counts.sum(axis=1, dtype=np.float64)
    prediction = (normalized * (library[:, None] / 1e4)).astype(np.float32)

    import scvi

    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = genes.tolist()
    metadata = {
        "method": "scvi",
        "method_version": str(scvi.__version__),
        "scale": "counts",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["development", "validation"],
        "training_data": {"cell_ids_sha256": order_hash(np.asarray(cell_ids)[training].tolist())},
        "parameters": {
            "n_hidden": 128, "n_latent": 10, "n_layers": 2, "gene_likelihood": "nb", "maximum_epochs": args.epochs,
            "extension": "new cells are encoded by the trained model",
            "fit_cells": int(training.sum()),
            "heldout_test_cells": int((split == "test").sum()),
            "test_used_for_fit": False,
            "fitting_cell_proposals_exclude_own_counts": True,
            "cross_fitting_folds": FOLDS,
            "condition_column": args.condition_column,
            "theta": "theta.npy row 0: model on all fitting cells; row 1 + f: model without partition f",
            "model_index": "model_index.npy: theta row of the model that proposed each cell",
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.input),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, prediction, metadata)
    np.save(Path(args.output) / "theta.npy", theta, allow_pickle=False)
    np.save(Path(args.output) / "model_index.npy", model_index, allow_pickle=False)
    print(json.dumps({"method": "scvi", "shape": list(prediction.shape),
                      "theta_median_per_model": np.nanmedian(theta, axis=1).round(4).tolist()}))

if __name__ == "__main__":
    main()
