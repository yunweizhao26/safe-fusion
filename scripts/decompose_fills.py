#!/usr/bin/env python3
"""Split a sparse fill into its masked-positive part and its recorded-zero part.

In the masked benchmark, the candidate zeros of test cells are either masked
positives (recorded nonzero counts hidden by the benchmark) or recorded zeros.
For each sparse output, this script writes two contracts with the same
selected entries restricted to one of the two groups. Every other entry keeps
the value of the source contract, so the two parts add up to the source fill.
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

from safefusion_benchmark.contracts import write_output_contract  # noqa: E402

PARTS = ("masked_only", "zeros_only")


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_source(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("sources must use name=contract_directory")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--source", action="append", required=True, help="name=sparse_contract_directory")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.corrupted)
    counts = dense(adata.layers["corrupted_counts"]).astype(np.float32)
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    split = (
        pd.read_parquet(args.splits).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    )
    test = split == "test"
    candidate = (counts == 0) & test[:, None]
    # The coordinates file stores the split used when the mask was created;
    # cross-fitting changes the split, so masked positives of test cells are
    # selected by the current split's cell indices.
    coordinates = pd.read_parquet(args.coordinates)
    masked = np.zeros(counts.shape, dtype=bool)
    masked[
        coordinates["cell_index"].to_numpy(dtype=int),
        coordinates["gene_index"].to_numpy(dtype=int),
    ] = True
    masked &= candidate
    recorded_zero = candidate & ~masked

    output_root = Path(args.output_root)
    summary: list[dict] = []
    for name, source in (parse_source(value) for value in args.source):
        mean = np.load(source / "mean.npy", allow_pickle=False)
        metadata = json.loads((source / "metadata.json").read_text())
        if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
            raise ValueError(f"cell or gene order differs for {source}")
        outside = ~candidate
        if not np.array_equal(mean[test][outside[test]], counts[test][outside[test]]):
            raise ValueError(f"{source} changes test entries that are not candidate zeros")
        changed = candidate & (mean != counts)
        for part, keep in (("masked_only", masked), ("zeros_only", recorded_zero)):
            output = mean.copy()
            drop = changed & ~keep
            output[drop] = counts[drop]
            write_output_contract(
                output_root / f"{name}__{part}",
                output,
                {
                    **metadata,
                    "method": f"{name}__{part}",
                    "parameters": {
                        **metadata.get("parameters", {}),
                        "decomposition": {
                            "part": part,
                            "source_contract": str(source),
                            "kept_entries": "selected masked positives" if part == "masked_only" else "selected recorded zeros",
                        },
                    },
                },
            )
        summary.append({
            "source": name,
            "n_candidates": int(candidate.sum()),
            "n_masked_candidates": int(masked.sum()),
            "n_changed": int(changed.sum()),
            "n_changed_masked": int((changed & masked).sum()),
            "n_changed_recorded_zero": int((changed & recorded_zero).sum()),
            "share_changed_masked": float((changed & masked).sum() / max(1, changed.sum())),
            "mean_inserted_masked": float(mean[changed & masked].mean()) if (changed & masked).any() else float("nan"),
            "mean_inserted_recorded_zero": float(mean[changed & recorded_zero].mean()) if (changed & recorded_zero).any() else float("nan"),
        })
        print(json.dumps(summary[-1]), flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(output_root / "decomposition_counts.csv", index=False)


if __name__ == "__main__":
    main()
