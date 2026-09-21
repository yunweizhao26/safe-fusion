#!/usr/bin/env python3
"""Run standard transductive scVI and emit zero-ranking expression scores."""

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

from safefusion_benchmark.contracts import order_hash, write_output_contract  # noqa: E402
from safefusion_benchmark.hashing import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    import scvi

    scvi.settings.seed = args.seed
    source = ad.read_h5ad(args.corrupted)
    matrix = source.layers["corrupted_counts"]
    counts = matrix.tocsr().astype(np.float32) if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    model_adata = ad.AnnData(X=counts, obs=source.obs.copy(), var=source.var.copy())
    model_adata.obs_names = source.obs_names.copy()
    model_adata.var_names = source.var_names.copy()

    scvi.model.SCVI.setup_anndata(model_adata)
    model = scvi.model.SCVI(
        model_adata,
        n_hidden=128,
        n_latent=10,
        n_layers=2,
        dispersion="gene",
        gene_likelihood="nb",
    )
    model.train(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        early_stopping=True,
        check_val_every_n_epoch=1,
    )
    expression = model.get_normalized_expression(
        model_adata,
        library_size=1e4,
        batch_size=256,
        return_mean=True,
    )
    mean = expression.to_numpy(dtype=np.float32) if hasattr(expression, "to_numpy") else np.asarray(expression, dtype=np.float32)

    split = (
        pd.read_parquet(args.splits)
        .set_index("cell_id")
        .loc[source.obs_names.astype(str), "split"]
        .to_numpy()
    )
    cell_ids = source.obs_names.astype(str).tolist()
    gene_ids = source.var_names.astype(str).tolist()
    metadata = {
        "method": "scvi",
        "method_version": str(scvi.__version__),
        "scale": "normalized_expression_1e4",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["transductive_full_corrupted_matrix"],
        "parameters": {
            "standard_usage": True,
            "transductive": True,
            "test_used_for_fit": True,
            "fit_cells": int(len(source)),
            "heldout_test_cells": int(np.sum(split == "test")),
            "n_hidden": 128,
            "n_latent": 10,
            "n_layers": 2,
            "gene_likelihood": "nb",
            "maximum_epochs": args.epochs,
            "epochs_trained": int(model.history["elbo_train"].shape[0]),
            "disclosure": "Standard scVI is fit to the full corrupted matrix without masking labels or original hidden values. It is a transductive sensitivity baseline, not a heldout projection result.",
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.corrupted),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, mean, metadata)
    model.save(str(Path(args.output) / "model"), overwrite=True, save_anndata=False)
    print(json.dumps({"method": "scvi", "shape": list(mean.shape), "parameters": metadata["parameters"]}))


if __name__ == "__main__":
    main()
