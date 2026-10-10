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

from safefusion_benchmark.corruption import corrupt_counts
from safefusion_benchmark.hashing import sha256_file


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(source)
    if "preassigned_split" not in adata.obs or "target" not in adata.obs:
        raise ValueError("Prepared Papalexi data lacks locked split or target fields")
    split = adata.obs["preassigned_split"].astype(str).to_numpy()
    if set(split) != {"development", "test"}:
        raise ValueError(f"Unexpected split values: {sorted(set(split))}")
    split_frame = pd.DataFrame(
        {"cell_id": adata.obs_names.astype(str), "split": split}
    )
    split_frame.to_parquet(output / "splits.parquet", index=False)

    counts = dense(adata.layers["counts"] if "counts" in adata.layers else adata.X).astype(np.int32)
    specification = {
        "kind": "stratified_nonzero_mask",
        "fraction": 0.10,
        "gene_bins": 4,
        "library_bins": 4,
    }
    corrupted, coordinates = corrupt_counts(
        counts,
        adata.obs_names.astype(str).to_numpy(),
        adata.var_names.astype(str).to_numpy(),
        adata.obs["target"].astype(str).to_numpy(),
        split,
        specification,
        args.seed,
    )
    corrupted_adata = adata.copy()
    corrupted_adata.X = sparse.csr_matrix(corrupted)
    for layer in list(corrupted_adata.layers):
        del corrupted_adata.layers[layer]
    corrupted_adata.layers["corrupted_counts"] = sparse.csr_matrix(corrupted)
    corrupted_adata.uns["corruption"] = {
        "name": "mask_010",
        "specification": specification,
        "seed": args.seed,
    }
    corrupted_adata.write_h5ad(output / "corrupted.h5ad", compression="gzip")
    coordinates.to_parquet(output / "coordinates.parquet", index=False)

    required = ["CD274", "CD86", "PDCD1LG2", "HAVCR2"]
    report = {
        "input": str(source),
        "input_sha256": sha256_file(source),
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "required_gene_presence": {
            gene: bool(gene in adata.var_names) for gene in required
        },
        "split_counts": pd.Series(split).value_counts().to_dict(),
        "masked_coordinates": int(len(coordinates)),
        "masked_coordinates_by_split": coordinates["split"].value_counts().to_dict(),
        "corruption": specification,
        "seed": args.seed,
    }
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
