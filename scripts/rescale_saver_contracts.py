#!/usr/bin/env python3
"""Put existing SAVER contracts on the count scale of their input.

``run_saver_baseline.R`` calls ``SAVER::saver`` with the default size factors
(each cell's library over the mean library), so the stored estimates are on the
normalized scale although the contracts recorded ``scale: counts``. This script
multiplies each estimate by its cell's size factor and each variance by the
squared size factor, as ``run_saver_baseline.py`` now does for new runs. It is
idempotent: contracts already marked ``count_scale_rescaled`` are skipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_saver_baseline import saver_size_factors  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--corrupted", type=Path, required=True, help="The corrupted h5ad given to SAVER.")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to rewriting the contract in place.")
    args = parser.parse_args()

    metadata = json.loads((args.contract / "metadata.json").read_text())
    if metadata.get("parameters", {}).get("count_scale_rescaled"):
        print(json.dumps({"contract": str(args.contract), "skipped": "already on the count scale"}))
        return
    adata = ad.read_h5ad(args.corrupted)
    if metadata["cell_ids"] != adata.obs_names.astype(str).tolist():
        raise ValueError("cell order differs between the contract and the corrupted input")
    if metadata["gene_ids"] != adata.var_names.astype(str).tolist():
        raise ValueError("gene order differs between the contract and the corrupted input")
    counts = adata.layers["corrupted_counts"]
    counts = counts.toarray() if sparse.issparse(counts) else np.asarray(counts)
    size_factor = saver_size_factors(counts.astype(np.float64))

    output = args.output or args.contract
    output.mkdir(parents=True, exist_ok=True)
    mean = np.load(args.contract / "mean.npy")
    np.save(output / "mean.npy", (mean * size_factor[:, None]).astype(np.float32))
    if (args.contract / "variance.npy").exists():
        variance = np.load(args.contract / "variance.npy")
        np.save(output / "variance.npy", (variance * np.square(size_factor)[:, None]).astype(np.float32))
    for extra in args.contract.iterdir():
        # Copy side files for a new output; the R work directory stays behind.
        if extra.is_file() and extra.name not in {"mean.npy", "variance.npy", "metadata.json"}:
            if not (output / extra.name).exists():
                shutil.copy2(extra, output / extra.name)
    metadata["parameters"]["count_scale_rescaled"] = True
    metadata["parameters"]["size_factor"] = "SAVER default (library / mean library); estimates multiplied by it"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "contract": str(args.contract),
        "output": str(output),
        "size_factor_range": [float(size_factor.min()), float(size_factor.max())],
    }))


if __name__ == "__main__":
    main()
