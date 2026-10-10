#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.hashing import sha256_file


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def with_counts(adata: ad.AnnData, counts: np.ndarray, layer_is_sparse: bool) -> ad.AnnData:
    result = adata.copy()
    result.layers["corrupted_counts"] = sparse.csr_matrix(counts) if layer_is_sparse else counts
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", required=True, help="h5ad with the recorded counts in layers['counts']")
    parser.add_argument("--corrupted", required=True, help="masked benchmark input")
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    if not np.array_equal(truth_adata.obs_names.astype(str), corrupted_adata.obs_names.astype(str)):
        raise ValueError("truth and corrupted cell order differ")
    if not np.array_equal(truth_adata.var_names.astype(str), corrupted_adata.var_names.astype(str)):
        raise ValueError("truth and corrupted gene order differ")
    recorded = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    layer_is_sparse = sparse.issparse(corrupted_adata.layers["corrupted_counts"])
    cell_ids = corrupted_adata.obs_names.astype(str)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = split == "test"
    if not test.any() or test.all():
        raise ValueError("the split file needs fitting and test cells")

    nonzero = corrupted > 0
    if not np.array_equal(corrupted[nonzero], recorded[nonzero]):
        raise ValueError("masked input is not a masked copy of the recorded counts")

    hybrid = corrupted.copy()
    hybrid[test] = recorded[test]
    coordinates = pd.read_parquet(args.coordinates)
    fitting_coordinates = coordinates[~test[coordinates["cell_index"].to_numpy(dtype=int)]].reset_index(drop=True)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with_counts(corrupted_adata, hybrid, layer_is_sparse).write_h5ad(output / "hybrid.h5ad")
    with_counts(corrupted_adata, recorded, layer_is_sparse).write_h5ad(output / "recorded.h5ad")
    fitting_coordinates.to_parquet(output / "coordinates.parquet", index=False)
    coordinates.iloc[:0].to_parquet(output / "empty_coordinates.parquet", index=False)
    shutil.copyfile(args.splits, output / "splits.parquet")

    test_zeros = int(np.sum(recorded[test] == 0))
    manifest = {
        "design": "fitting cells keep the benchmark mask; test cells use recorded counts",
        "truth": str(args.truth),
        "truth_sha256": sha256_file(args.truth),
        "corrupted": str(args.corrupted),
        "corrupted_sha256": sha256_file(args.corrupted),
        "coordinates": str(args.coordinates),
        "splits": str(args.splits),
        "n_cells": int(len(cell_ids)),
        "n_genes": int(recorded.shape[1]),
        "n_fitting_cells": int((~test).sum()),
        "n_test_cells": int(test.sum()),
        "n_fitting_masked_positives": int(len(fitting_coordinates)),
        "n_test_recorded_zeros": test_zeros,
        "test_recorded_zero_fraction": float(test_zeros / recorded[test].size),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
