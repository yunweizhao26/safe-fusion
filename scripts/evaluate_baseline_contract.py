#!/usr/bin/env python3
"""Evaluate a full matrix baseline on locked masked entries and test zeros."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method-output", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = ad.read_h5ad(args.corrupted)
    counts = dense(source.layers["corrupted_counts"]).astype(np.float32)
    method_output = Path(args.method_output)
    prediction = np.load(method_output / "mean.npy", mmap_mode="r", allow_pickle=False)
    metadata = json.loads((method_output / "metadata.json").read_text())
    cell_ids = source.obs_names.astype(str).tolist()
    gene_ids = source.var_names.astype(str).tolist()
    if list(prediction.shape) != list(counts.shape):
        raise ValueError("prediction shape differs from input shape")
    if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
        raise ValueError("cell or gene order differs from input")

    coordinates = pd.read_parquet(args.coordinates)
    if "split" in coordinates:
        coordinates = coordinates.loc[coordinates["split"].astype(str).eq("test")]
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    columns = coordinates["gene_index"].to_numpy(dtype=np.int64)
    truth = coordinates["original_value"].to_numpy(dtype=np.float64)
    values = np.asarray(prediction[rows, columns], dtype=np.float64)
    raw_mae = float(np.mean(np.log1p(truth)))
    method_mae = float(np.mean(np.abs(np.log1p(np.maximum(values, 0.0)) - np.log1p(truth))))

    split = (
        pd.read_parquet(args.splits)
        .set_index("cell_id")
        .loc[cell_ids, "split"]
        .astype(str)
        .to_numpy()
    )
    test_counts = counts[split == "test"]
    test_prediction = np.asarray(prediction[split == "test"], dtype=np.float32)
    zeros = test_counts == 0
    report = {
        "method": metadata["method"],
        "masked_entries": int(len(rows)),
        "raw_masked_log1p_mae": raw_mae,
        "method_masked_log1p_mae": method_mae,
        "masked_error_recovery": float(1.0 - method_mae / raw_mae),
        "test_zero_fill_fraction": float(np.mean(test_prediction[zeros] > 1e-8)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
