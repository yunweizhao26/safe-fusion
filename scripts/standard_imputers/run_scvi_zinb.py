#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--posterior-samples", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    import scvi
    import torch

    scvi.settings.seed = args.seed
    started = time.perf_counter()
    source = ad.read_h5ad(args.corrupted)
    matrix = source.layers["corrupted_counts"]
    counts = matrix.tocsr().astype(np.float32) if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))
    model_adata = ad.AnnData(X=counts, obs=source.obs.copy(), var=source.var.copy())
    model_adata.obs_names = source.obs_names.copy()
    model_adata.var_names = source.var_names.copy()

    scvi.model.SCVI.setup_anndata(model_adata)
    model = scvi.model.SCVI(model_adata, n_hidden=128, n_latent=10, n_layers=2, dispersion="gene", gene_likelihood="zinb")
    model.train(max_epochs=args.epochs, accelerator="gpu", devices=1, early_stopping=True, check_val_every_n_epoch=1)
    expression = model.get_normalized_expression(model_adata, library_size=1e4, batch_size=256, return_mean=True)
    mean = expression.to_numpy(dtype=np.float32) if hasattr(expression, "to_numpy") else np.asarray(expression, dtype=np.float32)

    module = model.module
    module.eval()
    posterior = np.zeros(counts.shape, dtype=np.float32)
    loader = model._make_data_loader(adata=model_adata, batch_size=args.batch_size, shuffle=False)
    start = 0
    with torch.inference_mode():
        for tensors in loader:
            _, generative = module.forward(tensors, inference_kwargs={"n_samples": args.posterior_samples}, compute_loss=False)
            px = generative["px"]
            mu, theta, pi = px.mu, px.theta, px.zi_probs
            theta = theta.expand_as(mu)
            nb_zero = torch.exp(theta * (torch.log(theta) - torch.log(theta + mu)))
            probability = pi / (pi + (1.0 - pi) * nb_zero)
            block = probability.mean(dim=0).cpu().numpy()
            posterior[start:start + block.shape[0]] = block
            start += block.shape[0]
    if start != counts.shape[0]:
        raise ValueError(f"posterior computed for {start} of {counts.shape[0]} cells")
    posterior[counts.toarray() > 0] = 0.0
    elapsed = time.perf_counter() - started

    split = pd.read_parquet(args.splits).set_index("cell_id").loc[source.obs_names.astype(str), "split"].to_numpy()
    cell_ids = source.obs_names.astype(str).tolist()
    gene_ids = source.var_names.astype(str).tolist()
    metadata = {
        "method": "scvi_zinb",
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
            "dispersion": "gene",
            "gene_likelihood": "zinb",
            "maximum_epochs": args.epochs,
            "early_stopping": True,
            "epochs_trained": int(model.history["elbo_train"].shape[0]),
            "posterior_samples": args.posterior_samples,
            "dropout_probability": "posterior probability of the zero-inflation component given a zero count, averaged over latent draws",
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "elapsed_seconds": elapsed,
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.corrupted),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, mean, metadata)
    np.save(Path(args.output) / "dropout_probability.npy", posterior, allow_pickle=False)
    print(json.dumps({"method": "scvi_zinb", "shape": list(mean.shape), "epochs": metadata["parameters"]["epochs_trained"],
                      "elapsed_seconds": round(elapsed, 1)}))

if __name__ == "__main__":
    main()
