#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.contracts import write_output_contract


def scvi_score(contract: Path, metadata: dict, library: np.ndarray) -> tuple[np.ndarray, dict]:
    import torch

    if metadata["scale"] != "normalized_expression_1e4":
        raise ValueError(f"expected normalized_expression_1e4, got {metadata['scale']}")
    state = torch.load(contract / "model" / "model.pt", map_location="cpu", weights_only=False)
    settings = state["attr_dict"]["init_params_"]["non_kwargs"]
    if settings["dispersion"] != "gene" or settings["gene_likelihood"] != "nb":
        raise ValueError(f"expected gene dispersion and a negative binomial, got {settings}")
    if list(state["var_names"]) != metadata["gene_ids"]:
        raise ValueError("gene order of the saved model differs from the contract")
    theta = np.exp(state["model_state_dict"]["px_r"].numpy().astype(np.float64))
    mu = np.load(contract / "mean.npy").astype(np.float64) * (library[:, None] / 1e4)
    score = theta[None, :] * np.log1p(mu / theta[None, :])
    return score, {
        "score": "-log P(X = 0) = theta log1p(mu / theta) under the negative binomial of scVI",
        "theta": "exp(px_r) of the saved model, one value per gene",
        "theta_quantiles": {str(q): float(np.quantile(theta, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
    }


def saver_score(contract: Path, metadata: dict) -> tuple[np.ndarray, dict]:
    if metadata["scale"] != "counts" or not metadata["parameters"].get("count_scale_rescaled"):
        raise ValueError("expected a SAVER contract rescaled to counts")
    mean = np.maximum(np.load(contract / "mean.npy").astype(np.float64), 0.0)
    variance = np.maximum(np.load(contract / "variance.npy").astype(np.float64), 0.0)
    positive = (mean > 0) & (variance > 0)
    ratio = np.divide(variance, mean, out=np.zeros_like(mean), where=positive)
    shape = np.divide(np.square(mean), variance, out=np.zeros_like(mean), where=positive)
    score = np.where(positive, shape * np.log1p(ratio), mean)
    return score, {
        "score": "-log P(X = 0) = (m^2 / v) log1p(v / m) under the gamma-Poisson posterior predictive of SAVER",
        "mean_and_variance": "SAVER estimate and squared standard error, both on the count scale of the contract",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["scvi", "saver"], required=True)
    parser.add_argument("--contract", type=Path, required=True, help="Standard scVI contract with model/model.pt, or a SAVER contract with variance.npy.")
    parser.add_argument("--corrupted", type=Path, required=True, help="Masked input the model was fitted on.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.contract / "metadata.json").read_text())
    adata = ad.read_h5ad(args.corrupted)
    if metadata["cell_ids"] != adata.obs_names.astype(str).tolist():
        raise ValueError("cell order of the contract differs from the masked input")
    if metadata["gene_ids"] != adata.var_names.astype(str).tolist():
        raise ValueError("gene order of the contract differs from the masked input")
    layer = adata.layers["corrupted_counts"]
    counts = layer.toarray() if sparse.issparse(layer) else np.asarray(layer)
    library = counts.astype(np.float32).sum(axis=1, dtype=np.float64)
    if args.method == "scvi":
        score, details = scvi_score(args.contract, metadata, library)
    else:
        score, details = saver_score(args.contract, metadata)
    output = {
        **metadata,
        "method": f"{args.method}_nonzero_probability",
        "scale": "negative_log_zero_probability",
        "source_contract": str(args.contract),
        "parameters": {**metadata["parameters"], **details},
    }
    write_output_contract(args.output, score.astype(np.float32), output)
    print(json.dumps({"output": str(args.output), "shape": list(score.shape), **details}))


if __name__ == "__main__":
    main()
