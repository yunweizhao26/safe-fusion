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
from sklearn.utils.extmath import randomized_svd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.contracts import write_output_contract


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def normalize_data(counts: np.ndarray) -> np.ndarray:
    total = counts.sum(axis=1)
    safe_total = np.where(total > 0, total, 1.0)
    return np.log1p(counts / safe_total[:, None] * 1e4).astype(np.float32)


def choose_k(singular_values: np.ndarray, thresh: float = 6.0, noise_start: int = 80) -> int:
    values = singular_values[: max(1, len(singular_values) - 1)]
    diffs = values[:-1] - values[1:]
    noise_start = min(noise_start, len(diffs))
    if noise_start < 1:
        return min(20, len(diffs))
    noise_diffs = diffs[noise_start - 1 :]
    mu = float(np.mean(noise_diffs))
    sigma = float(np.std(noise_diffs))
    if sigma == 0:
        return min(20, len(diffs))
    num_of_sds = (diffs - mu) / sigma
    candidates = np.flatnonzero(num_of_sds > thresh)
    return int(candidates.max() + 1) if len(candidates) else min(20, len(diffs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--components", type=int, default=100)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--quantile-prob", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    corrupted_adata = ad.read_h5ad(args.corrupted)
    counts = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        corrupted_adata.obs_names.astype(str), "split"
    ].to_numpy()
    train = np.isin(split, ["development", "validation"])
    test = split == "test"

    normalized = normalize_data(counts)
    train_normalized = normalized[train]
    k_max = min(args.components, train_normalized.shape[0] - 1, train_normalized.shape[1] - 1)
    u, d, v = randomized_svd(
        train_normalized,
        n_components=k_max,
        n_iter=args.power_iterations,
        random_state=args.seed,
    )
    k = choose_k(d)
    v_k = v[:k].T
    rank_k = (normalized @ v_k) @ v_k.T

    thresholds = np.abs(np.quantile(rank_k[train], args.quantile_prob, axis=0))
    rank_cor = np.where(rank_k <= thresholds[None, :], 0.0, rank_k)

    def nonzero_sd(matrix: np.ndarray) -> np.ndarray:
        result = np.zeros(matrix.shape[1])
        for gene in range(matrix.shape[1]):
            values = matrix[:, gene]
            nonzero = values[values != 0]
            result[gene] = nonzero.std() if len(nonzero) else 0.0
        return result

    sigma_1 = nonzero_sd(rank_cor[train])
    sigma_2 = nonzero_sd(normalized[train])
    nonzero_counts_1 = (rank_cor[train] != 0).sum(axis=0)
    nonzero_counts_2 = (normalized[train] != 0).sum(axis=0)
    mu_1 = np.divide(rank_cor[train].sum(axis=0), nonzero_counts_1, out=np.zeros_like(sigma_1), where=nonzero_counts_1 > 0)
    mu_2 = np.divide(normalized[train].sum(axis=0), nonzero_counts_2, out=np.zeros_like(sigma_2), where=nonzero_counts_2 > 0)

    toscale = ~np.isnan(sigma_1) & ~np.isnan(sigma_2) & ~((sigma_1 == 0) & (sigma_2 == 0)) & (sigma_1 != 0)
    sigma_ratio = np.divide(sigma_2, sigma_1, out=np.ones_like(sigma_1), where=sigma_1 != 0)
    to_add = -mu_1 * np.divide(sigma_2, sigma_1, out=np.zeros_like(sigma_1), where=sigma_1 != 0) + mu_2
    scaled = rank_cor * sigma_ratio[None, :] + to_add[None, :]
    scaled[rank_cor == 0] = 0.0
    scaled[scaled < 0] = 0.0
    originally_nonzero = normalized > 0
    scaled[originally_nonzero & (scaled == 0)] = normalized[originally_nonzero & (scaled == 0)]

    library = counts.sum(axis=1)
    safe_library = np.where(library > 0, library, 1.0)
    predicted_counts = np.expm1(scaled) * (safe_library[:, None] / 1e4)
    predicted_counts = np.clip(predicted_counts, 0.0, None).astype(np.float32)

    output = Path(args.output)
    metadata = {
        "method": "alra",
        "method_version": "official KlugerLab ALRA (R package), python port",
        "scale": "counts",
        "cell_ids": corrupted_adata.obs_names.astype(str).tolist(),
        "gene_ids": corrupted_adata.var_names.astype(str).tolist(),
        "seed": args.seed,
        "parameters": {
            "fit_cells": int(train.sum()),
            "heldout_test_cells": int(test.sum()),
            "test_used_for_fit": False,
            "chosen_k": k,
            "power_iterations": args.power_iterations,
            "quantile_prob": args.quantile_prob,
            "adaptation": "SVD, per-gene thresholds and scaling fit on training cells only; test cells projected through fitted components",
        },
    }
    write_output_contract(output, predicted_counts, metadata)
    print(json.dumps({"method": "alra", "chosen_k": k, "shape": list(predicted_counts.shape), "parameters": metadata["parameters"]}))


if __name__ == "__main__":
    main()
