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

from safefusion_benchmark.corruption import corrupt_counts


def split_within_conditions(conditions: np.ndarray, test_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    split = np.empty(len(conditions), dtype=object)
    for condition in sorted(set(conditions)):
        positions = np.flatnonzero(conditions == condition)
        permuted = rng.permutation(positions)
        n_test = int(round(len(positions) * test_fraction))
        split[permuted[:n_test]] = "test"
        split[permuted[n_test:]] = "development"
    return split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--mask-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--split-column",
        default=None,
        help="Use a preassigned development/test column instead of drawing a new split.",
    )
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    counts = (adata.layers["counts"].toarray() if sparse.issparse(adata.layers["counts"]) else np.asarray(adata.layers["counts"])).astype(np.float32)
    conditions = adata.obs["condition"].astype(str).to_numpy()
    cell_ids = adata.obs_names.astype(str).to_numpy()
    gene_ids = adata.var_names.astype(str).to_numpy()

    if args.split_column:
        if args.split_column not in adata.obs:
            raise KeyError(f"missing split column: {args.split_column}")
        split = adata.obs[args.split_column].astype(str).to_numpy()
        if not set(split).issubset({"development", "test"}):
            raise ValueError("preassigned split must contain only development/test")
    else:
        split = split_within_conditions(conditions, args.test_fraction, args.seed)
    if np.any(split == ""):
        raise AssertionError("every cell must receive a split")

    split_frame = pd.DataFrame({
        "cell_id": cell_ids,
        "biological_unit": conditions,
        "condition": conditions,
        "split": split,
    })
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    split_path = destination / "splits.parquet"
    split_frame.to_parquet(split_path, index=False)

    corrupted, coordinates = corrupt_counts(
        counts,
        cell_ids,
        gene_ids,
        conditions,
        split,
        {"kind": "stratified_nonzero_mask", "fraction": args.mask_fraction, "gene_bins": 4, "library_bins": 4},
        args.seed,
    )
    corrupted_adata = ad.AnnData(
        X=sparse.csr_matrix(corrupted),
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )
    corrupted_adata.obs_names = adata.obs_names
    corrupted_adata.var_names = adata.var_names
    corrupted_adata.layers["corrupted_counts"] = sparse.csr_matrix(corrupted)
    corrupted_adata.write_h5ad(destination / "corrupted.h5ad")
    coordinates.to_parquet(destination / "coordinates.parquet", index=False)

    report = {
        "input": args.input,
        "cells": int(len(cell_ids)),
        "genes": int(len(gene_ids)),
        "conditions": sorted(set(conditions)),
        "n_conditions": int(len(set(conditions))),
        "test_fraction": args.test_fraction,
        "mask_fraction": args.mask_fraction,
        "masked_entries": int(len(coordinates)),
        "split_counts": split_frame["split"].value_counts().to_dict(),
        "seed": args.seed,
        "split_source": args.split_column or "deterministic stratified draw",
    }
    (destination / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
