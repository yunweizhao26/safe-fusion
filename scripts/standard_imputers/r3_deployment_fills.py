#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from compute_matched_baseline_f1_curves import tie_broken_order
from masked_f1_units import COUNT_SCALE, dense, stored_scale
from safefusion_benchmark.contracts import write_output_contract

def fraction_suffix(fraction: float) -> str:
    return f"{fraction:g}".replace(".", "p")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--name", required=True, help="output name prefix, for example scimpute or scimpute_dropout")
    parser.add_argument("--ranking", choices=["value", "dropout"], default="value")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--tie-seed", type=int, default=1729)
    args = parser.parse_args()

    hybrid = ad.read_h5ad(args.hybrid)
    recorded_adata = ad.read_h5ad(args.recorded)
    cell_ids = hybrid.obs_names.astype(str).tolist()
    gene_ids = hybrid.var_names.astype(str).tolist()
    if recorded_adata.obs_names.astype(str).tolist() != cell_ids or recorded_adata.var_names.astype(str).tolist() != gene_ids:
        raise ValueError("recorded and hybrid inputs differ in cell or gene order")
    counts = dense(hybrid.layers["corrupted_counts"]).astype(np.float32)
    recorded = dense(recorded_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = split == "test"
    if not np.array_equal(counts[test], recorded[test]):
        raise ValueError("the held-out cells of the hybrid input do not hold their recorded counts")
    rows, cols = np.where((recorded == 0) & test[:, None])

    metadata = json.loads((args.contract / "metadata.json").read_text())
    if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
        raise ValueError(f"cell or gene order differs for {args.contract}")
    scale = stored_scale(metadata)
    library = counts.sum(axis=1, dtype=np.float64)
    mean = np.load(args.contract / "mean.npy", mmap_mode="r")
    value = np.maximum(COUNT_SCALE[scale](np.asarray(mean[rows, cols], dtype=np.float64), library[rows]), 0.0)
    if args.ranking == "value":
        score = value
    else:
        score = np.asarray(np.load(args.contract / "dropout_probability.npy", mmap_mode="r")[rows, cols], dtype=np.float64)
    order = tie_broken_order(score, args.tie_seed)

    summary = []
    for fraction in args.fractions:
        k = max(1, int(round(fraction * len(rows))))
        chosen = order[:k]
        output = recorded.copy()
        output[rows[chosen], cols[chosen]] = value[chosen].astype(np.float32)
        name = f"{args.name}_topk_{fraction_suffix(fraction)}"
        changed = int(np.sum(value[chosen] > 0))
        write_output_contract(args.output_root / name, output, {
            **metadata,
            "method": name,
            "scale": "counts",
            "parameters": {
                **metadata.get("parameters", {}),
                "deployment_fill": {
                    "fill_fraction": fraction, "ranking": args.ranking, "tie_seed": args.tie_seed,
                    "source_contract": str(args.contract), "source_scale": scale,
                    "candidates": "recorded zeros of the held-out cells", "n_candidates": int(len(rows)),
                    "n_selected": int(k), "n_changed": changed, "non_test_cells": "recorded counts",
                },
            },
        })
        summary.append({"name": name, "fraction": fraction, "n_candidates": int(len(rows)), "n_selected": int(k),
                        "n_changed": changed, "mean_inserted_value": float(value[chosen][value[chosen] > 0].mean()) if changed else float("nan")})
        print(json.dumps(summary[-1]), flush=True)
    path = args.output_root / "fill_counts.csv"
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(previous):
        previous = previous[~previous["name"].isin([row["name"] for row in summary])]
    pd.concat([previous, pd.DataFrame(summary)], ignore_index=True).to_csv(path, index=False)

if __name__ == "__main__":
    main()
