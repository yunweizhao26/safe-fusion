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

from safefusion_benchmark.contracts import write_output_contract

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit-dir", type=Path, required=True, help="Deployment unit with recorded.h5ad, splits.parquet and safe_fusion_<pct>pct.")
    parser.add_argument("--value-contract", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Output name prefix, for example safe_fusion_rate_poisson.")
    parser.add_argument("--pcts", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.unit_dir / "recorded.h5ad")
    matrix = adata.layers["corrupted_counts"]
    recorded = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    split = pd.read_parquet(args.unit_dir / "splits.parquet").set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = split == "test"
    value_metadata = json.loads((args.value_contract / "metadata.json").read_text())
    if value_metadata["cell_ids"] != cell_ids or value_metadata["gene_ids"] != gene_ids:
        raise ValueError(f"cell or gene order differs for {args.value_contract}")
    value = np.load(args.value_contract / "mean.npy", mmap_mode="r")

    rows = []
    for pct in args.pcts:
        source = args.unit_dir / f"safe_fusion_{pct}pct"
        metadata = json.loads((source / "metadata.json").read_text())
        if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
            raise ValueError(f"cell or gene order differs for {source}")
        current = np.load(source / "mean.npy")
        if not np.array_equal(current[~test], recorded[~test]) or not np.array_equal(current[recorded > 0], recorded[recorded > 0]):
            raise ValueError(f"{source} changes entries outside the recorded zeros of the held-out cells")
        selected = (recorded == 0) & (current != recorded) & test[:, None]
        cell, gene = np.nonzero(selected)
        output = recorded.copy()
        output[cell, gene] = np.maximum(np.asarray(value[cell, gene], dtype=np.float32), 0.0)
        name = f"{args.name}_{pct}pct"
        write_output_contract(args.output_root / name, output, {
            **metadata,
            "method": name,
            "parameters": {
                **metadata.get("parameters", {}),
                "inserted_value": {
                    "value_contract": str(args.value_contract),
                    "positions": f"recorded zeros of the held-out cells that {source} fills",
                },
            },
        })
        inserted = output[cell, gene]
        rows.append({
            "name": name, "fill_pct": pct, "selected": int(len(cell)),
            "test_recorded_zeros": int(((recorded == 0) & test[:, None]).sum()),
            "current_mean": float(current[cell, gene].mean()), "current_median": float(np.median(current[cell, gene])),
            "new_mean": float(inserted.mean()), "new_median": float(np.median(inserted)),
            "new_share_below_one": float((inserted < 1.0).mean()), "new_share_zero": float((inserted <= 0.0).mean()),
            "current_mean_log1p": float(np.log1p(current[cell, gene]).mean()), "new_mean_log1p": float(np.log1p(inserted).mean()),
        })
        print(json.dumps(rows[-1]), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_root / f"fill_counts_{args.name}.csv", index=False)

if __name__ == "__main__":
    main()
