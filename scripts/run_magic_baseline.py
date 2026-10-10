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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    import magic

    source = ad.read_h5ad(args.corrupted)
    matrix = source.layers["corrupted_counts"]
    counts = matrix.toarray().astype(np.float32) if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float32)
    library = counts.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    normalized = np.log1p(counts * scale[:, None]).astype(np.float32)
    operator = magic.MAGIC(random_state=args.seed, n_jobs=args.n_jobs, verbose=1)
    mean = np.asarray(operator.fit_transform(normalized), dtype=np.float32)

    split = (
        pd.read_parquet(args.splits)
        .set_index("cell_id")
        .loc[source.obs_names.astype(str), "split"]
        .to_numpy()
    )
    cell_ids = source.obs_names.astype(str).tolist()
    gene_ids = source.var_names.astype(str).tolist()
    metadata = {
        "method": "magic",
        "method_version": str(getattr(magic, "__version__", "3.0.0")),
        "scale": "log1p_cpm",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["transductive_full_corrupted_matrix"],
        "parameters": {
            "standard_defaults": True,
            "transductive": True,
            "test_used_for_fit": True,
            "fit_cells": int(len(source)),
            "heldout_test_cells": int(np.sum(split == "test")),
            "n_jobs": args.n_jobs,
            "disclosure": "Standard MAGIC is fit to the full corrupted matrix without masking labels or original hidden values. It is a transductive sensitivity baseline, not a heldout projection result.",
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.corrupted),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, mean, metadata)
    print(json.dumps({"method": "magic", "shape": list(mean.shape), "parameters": metadata["parameters"]}))


if __name__ == "__main__":
    main()
