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
sys.path.insert(0, str(REPOSITORY / "scripts"))

from safefusion_benchmark.contracts import write_output_contract
from selector_attribution import exact_topk


def fraction_suffix(fraction: float) -> str:
    return f"{fraction:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method-contract", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[round(0.01 * pct, 2) for pct in range(1, 11)],
    )
    args = parser.parse_args()

    adata = ad.read_h5ad(args.corrupted)
    counts = adata.layers["corrupted_counts"]
    counts = (counts.toarray() if sparse.issparse(counts) else np.asarray(counts)).astype(np.float32)
    split = (
        pd.read_parquet(args.splits)
        .set_index("cell_id")
        .loc[adata.obs_names.astype(str), "split"]
        .to_numpy()
    )
    rows, cols = np.where((counts == 0) & (split == "test")[:, None])

    source = Path(args.method_contract)
    imputed = np.load(source / "mean.npy", allow_pickle=False)
    metadata = json.loads((source / "metadata.json").read_text())
    if metadata["cell_ids"] != adata.obs_names.astype(str).tolist():
        raise ValueError(f"cell order differs for {source}")
    if metadata["gene_ids"] != adata.var_names.astype(str).tolist():
        raise ValueError(f"gene order differs for {source}")
    scores = np.maximum(imputed[rows, cols], 0.0)

    for fraction in args.fractions:
        selected = exact_topk(scores, max(1, int(round(fraction * len(scores)))))
        output = counts.copy()
        output[rows[selected], cols[selected]] = scores[selected]
        name = f"{args.method_name}_topk_{fraction_suffix(fraction)}"
        write_output_contract(
            Path(args.output_root) / name,
            output,
            {
                **metadata,
                "method": name,
                "parameters": {
                    **metadata.get("parameters", {}),
                    "fill_fraction": fraction,
                    "ranking_score": "imputed_value",
                    "source_contract": str(source),
                },
            },
        )
        print(f"{name}: filled {int(selected.sum())} of {len(scores)} test zeros")


if __name__ == "__main__":
    main()
