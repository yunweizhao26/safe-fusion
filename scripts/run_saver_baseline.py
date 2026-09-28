#!/usr/bin/env python3
"""Run official SAVER on a corrupted count matrix and emit a contract.

SAVER has no held-out projection API, so it is run on the full matrix in its
standard usage; the contract metadata declares test cells were used for
fitting, and SAVER is reported only as a standard-usage baseline, excluded
from leakage-audited primary comparisons.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread, mmwrite


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.contracts import write_output_contract  # noqa: E402


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def saver_size_factors(counts: np.ndarray) -> np.ndarray:
    """SAVER's default size factors: each cell's library over the mean library."""
    library = counts.sum(axis=1, dtype=np.float64)
    return library / library.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rscript", default=str(REPOSITORY / ".conda-r-baselines/bin/Rscript"))
    parser.add_argument("--r-script", default=str(REPOSITORY / "scripts/run_saver_baseline.R"))
    parser.add_argument("--ncores", type=int, default=1)
    args = parser.parse_args()

    corrupted_adata = ad.read_h5ad(args.corrupted)
    counts = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        corrupted_adata.obs_names.astype(str), "split"
    ].to_numpy()
    work = Path(args.output) / "work"
    work.mkdir(parents=True, exist_ok=True)

    counts_path = work / "counts.mtx"
    mmwrite(counts_path, sparse.csr_matrix(counts))
    estimate_path = work / "estimate.mtx"
    se_path = work / "se.mtx"
    subprocess.run(
        [
            args.rscript,
            args.r_script,
            str(counts_path),
            str(estimate_path),
            str(se_path),
            str(args.ncores),
        ],
        check=True,
    )
    estimate = np.asarray(mmread(estimate_path).todense())
    variance = np.asarray(mmread(se_path).todense()) ** 2
    # SAVER's default size factors (library / mean library) return estimates
    # on the normalized scale; multiply back to the count scale of the input.
    size_factor = saver_size_factors(counts)
    estimate = np.clip(estimate * size_factor[:, None], 0.0, None).astype(np.float32)
    variance = np.clip(variance * np.square(size_factor)[:, None], 0.0, None).astype(np.float32)

    output = Path(args.output)
    metadata = {
        "method": "saver",
        "method_version": "official SAVER R package",
        "scale": "counts",
        "cell_ids": corrupted_adata.obs_names.astype(str).tolist(),
        "gene_ids": corrupted_adata.var_names.astype(str).tolist(),
        "parameters": {
            "fit_cells": int(np.sum(np.isin(split, ["development", "validation"]))),
            "heldout_test_cells": int(np.sum(split == "test")),
            "test_used_for_fit": True,
            "ncores": args.ncores,
            "count_scale_rescaled": True,
            "size_factor": "SAVER default (library / mean library); estimates multiplied by it",
            "disclosure": "SAVER has no held-out projection API; run on the full corrupted matrix (standard usage). Reported only as a standard-usage baseline and excluded from leakage-audited primary comparisons.",
        },
    }
    write_output_contract(output, estimate, metadata, variance=variance)
    print(json.dumps({"method": "saver", "shape": list(estimate.shape), "parameters": metadata["parameters"]}))


if __name__ == "__main__":
    main()
