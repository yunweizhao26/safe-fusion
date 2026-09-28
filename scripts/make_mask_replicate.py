#!/usr/bin/env python3
"""Draw one replicate of the stratified 10% nonzero mask with a new seed.

The replicate uses the production corruption (``corrupt_counts`` with the
stratified nonzero mask) and writes a masked input with the same layout as the
production files: ``layers["corrupted_counts"]`` holds the masked counts, and
the coordinates parquet lists every hidden entry. Only the seed differs.
"""

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

from safefusion_benchmark.corruption import corrupt_counts  # noqa: E402

SPEC = {"kind": "stratified_nonzero_mask", "fraction": 0.10, "gene_bins": 4, "library_bins": 4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, help="h5ad with the unmasked counts layer")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--unit-column", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    truth = ad.read_h5ad(args.truth)
    matrix = truth.layers["counts"] if "counts" in truth.layers else truth.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.int32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[truth.obs_names.astype(str), "split"].to_numpy()
    corrupted, coordinates = corrupt_counts(
        counts,
        truth.obs_names.astype(str).to_numpy(),
        truth.var_names.astype(str).to_numpy(),
        truth.obs[args.unit_column].astype(str).to_numpy(),
        split,
        SPEC,
        args.seed,
    )
    output = ad.AnnData(X=sparse.csr_matrix(corrupted), obs=truth.obs.copy(), var=truth.var.copy())
    output.layers["corrupted_counts"] = sparse.csr_matrix(corrupted)
    output.uns["corruption"] = {"name": "mask_010", "spec": SPEC, "seed": args.seed}
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    output.write_h5ad(destination / "corrupted.h5ad")
    coordinates.to_parquet(destination / "coordinates.parquet", index=False)
    report = {"truth": args.truth, "seed": args.seed, "spec": SPEC, "masked_entries": int(len(coordinates)),
              "cells": int(counts.shape[0]), "genes": int(counts.shape[1])}
    (destination / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
