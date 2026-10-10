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


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_source(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("sources must use name=contract_directory")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded", required=True, help="recorded.h5ad from build_deployment_inputs.py")
    parser.add_argument("--splits", required=True)
    parser.add_argument("--source", action="append", required=True, help="name=contract_directory")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.recorded)
    recorded = dense(adata.layers["corrupted_counts"]).astype(np.float32)
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = split == "test"
    rows: list[dict] = []
    for name, source in (parse_source(value) for value in args.source):
        mean = np.load(source / "mean.npy", allow_pickle=False)
        metadata = json.loads((source / "metadata.json").read_text())
        if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
            raise ValueError(f"cell or gene order differs for {source}")
        output = recorded.copy()
        output[test] = mean[test]
        changed = (output[test] != recorded[test])
        zeros = recorded[test] == 0
        write_output_contract(
            Path(args.output_root) / name,
            output,
            {
                **metadata,
                "method": name,
                "parameters": {
                    **metadata.get("parameters", {}),
                    "deployment": {
                        "source_contract": str(source),
                        "non_test_cells": "recorded counts",
                    },
                },
            },
        )
        rows.append({
            "method": name,
            "n_test_recorded_zeros": int(zeros.sum()),
            "n_changed_zeros": int((changed & zeros).sum()),
            "n_changed_nonzeros": int((changed & ~zeros).sum()),
            "changed_zero_fraction": float((changed & zeros).sum() / max(1, zeros.sum())),
            "mean_inserted_value": float(output[test][changed & zeros].mean()) if (changed & zeros).any() else float("nan"),
        })
        print(json.dumps(rows[-1]), flush=True)
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Path(args.output_root) / "deployment_fill_counts.csv", index=False)


if __name__ == "__main__":
    main()
