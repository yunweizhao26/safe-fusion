#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import COUNT_SCALE, dense
from safefusion_benchmark.contracts import write_output_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True, help="Source contract directory (mean.npy, metadata.json).")
    parser.add_argument("--corrupted", type=Path, required=True, help="Masked input whose library sizes set the count scale.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads((args.contract / "metadata.json").read_text())
    adata = ad.read_h5ad(args.corrupted)
    if metadata["cell_ids"] != adata.obs_names.astype(str).tolist():
        raise ValueError("cell order of the contract differs from the masked input")
    if metadata["gene_ids"] != adata.var_names.astype(str).tolist():
        raise ValueError("gene order of the contract differs from the masked input")
    scale = metadata["scale"]
    if scale not in COUNT_SCALE:
        raise ValueError(f"no count-scale rule for scale {scale!r}")
    library = dense(adata.layers["corrupted_counts"]).astype(np.float32).sum(axis=1, dtype=np.float64)
    mean = np.load(args.contract / "mean.npy", allow_pickle=False).astype(np.float64)
    values = COUNT_SCALE[scale](mean, library[:, None]).astype(np.float32)
    converted = {
        **metadata,
        "scale": "counts",
        "source_contract": str(args.contract),
        "source_scale": scale,
    }
    write_output_contract(args.output, values, converted)
    print(json.dumps({"source": str(args.contract), "source_scale": scale, "output": str(args.output), "shape": list(values.shape)}))


if __name__ == "__main__":
    main()
