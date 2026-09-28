#!/usr/bin/env python3
"""Prepare a binomial-thinning benchmark input.

Writes ``corrupted.h5ad`` (thinned counts in ``layers["corrupted_counts"]``)
and ``coordinates.parquet``. The coordinates list only the entries that are
nonzero in the unthinned counts and zero after thinning; these are the
positives of the benchmark, and ``original_value`` holds the unthinned count.
Either reuse an existing thinned matrix (``--corrupted``) or thin the counts
with ``--retained-fraction`` and ``--seed``.
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
from safefusion_benchmark.hashing import sha256_file  # noqa: E402


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True, help="Unthinned counts in layers['counts'].")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corrupted", default=None, help="Existing thinned matrix to reuse.")
    parser.add_argument("--retained-fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--unit-column", default="donor")
    parser.add_argument("--split-column", default="locked_split")
    args = parser.parse_args()
    if (args.corrupted is None) == (args.retained_fraction is None):
        raise ValueError("give exactly one of --corrupted and --retained-fraction")

    truth = ad.read_h5ad(args.truth)
    counts = dense(truth.layers["counts"])
    if not np.array_equal(counts, np.round(counts)):
        raise ValueError("truth counts are not integers")
    counts = counts.astype(np.int64)
    cell_ids = truth.obs_names.astype(str).to_numpy()
    gene_ids = truth.var_names.astype(str).to_numpy()
    units = truth.obs[args.unit_column].astype(str).to_numpy()
    splits = truth.obs[args.split_column].astype(str).to_numpy()

    if args.corrupted is not None:
        source = ad.read_h5ad(args.corrupted)
        if not (np.array_equal(source.obs_names.astype(str), cell_ids) and np.array_equal(source.var_names.astype(str), gene_ids)):
            raise ValueError("corrupted and truth cell or gene order differ")
        thinned = dense(source.layers["corrupted_counts"]).astype(np.int64)
        if np.any(thinned > counts) or np.any(thinned < 0):
            raise ValueError("corrupted matrix is not a thinning of the truth counts")
        obs, var = source.obs.copy(), source.var.copy()
        corruption = {"source": str(args.corrupted), "source_sha256": sha256_file(args.corrupted)}
    else:
        spec = {"kind": "binomial_thinning", "retained_fraction": float(args.retained_fraction)}
        thinned, _ = corrupt_counts(counts, cell_ids, gene_ids, units, splits, spec, args.seed)
        thinned = thinned.astype(np.int64)
        obs, var = truth.obs.copy(), truth.var.copy()
        corruption = {**spec, "seed": args.seed}

    rows, cols = np.where((counts > 0) & (thinned == 0))
    coordinates = pd.DataFrame({
        "cell_id": cell_ids[rows],
        "gene_id": gene_ids[cols],
        "cell_index": rows.astype(np.int64),
        "gene_index": cols.astype(np.int64),
        "corruption_type": "binomial_thinning_zeroed",
        "original_value": counts[rows, cols].astype(np.float32),
        "corrupted_value": np.zeros(len(rows), dtype=np.float32),
        "biological_unit": units[rows],
        "split": splits[rows],
    })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    matrix = sparse.csr_matrix(thinned.astype(np.float32))
    result = ad.AnnData(X=matrix, obs=obs, var=var)
    result.layers["corrupted_counts"] = matrix.copy()
    result.uns["corruption"] = json.dumps(corruption)
    result.write_h5ad(output / "corrupted.h5ad")
    coordinates.to_parquet(output / "coordinates.parquet", index=False)
    manifest = {
        "truth": str(args.truth),
        "truth_sha256": sha256_file(args.truth),
        "corruption": corruption,
        "shape": list(counts.shape),
        "entries_set_to_zero": int(len(coordinates)),
        "original_count_1_share": float(np.mean(coordinates["original_value"] == 1)) if len(coordinates) else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
