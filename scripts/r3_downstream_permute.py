#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.hashing import sha256_file

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def permute_within_groups(counts: np.ndarray, rows: np.ndarray, groups: np.ndarray, seed: int) -> np.ndarray:

    result = counts.copy()
    rng = np.random.default_rng(seed)
    for level in sorted(np.unique(groups)):
        members = rows[groups == level]
        order = np.argsort(rng.random((len(members), counts.shape[1])), axis=0)
        result[members] = np.take_along_axis(counts[members], order, axis=0)
    return result

def write_counts(template: ad.AnnData, counts: np.ndarray, path: Path) -> None:
    result = template.copy()
    matrix = sparse.csr_matrix(counts.astype(np.float32))
    result.X = matrix
    result.layers["corrupted_counts"] = matrix.copy()
    result.write_h5ad(path)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deployment-dir", type=Path, required=True, help="unit directory of the deployment analysis")
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.deployment_dir
    hybrid_adata = ad.read_h5ad(source / "hybrid.h5ad")
    recorded_adata = ad.read_h5ad(source / "recorded.h5ad")
    if not np.array_equal(hybrid_adata.obs_names, recorded_adata.obs_names) or not np.array_equal(hybrid_adata.var_names, recorded_adata.var_names):
        raise ValueError("hybrid and recorded inputs differ in cell or gene order")
    hybrid = dense(hybrid_adata.layers["corrupted_counts"]).astype(np.float32)
    recorded = dense(recorded_adata.layers["corrupted_counts"]).astype(np.float32)
    cell_ids = recorded_adata.obs_names.astype(str)
    split = pd.read_parquet(source / "splits.parquet").set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    test = split == "test"
    if not np.array_equal(hybrid[test], recorded[test]):
        raise ValueError("test rows of the deployment hybrid are not the recorded counts")

    test_rows = np.flatnonzero(test)
    labels = recorded_adata.obs[args.label_column].astype(str).to_numpy()
    permuted = permute_within_groups(recorded, test_rows, labels[test_rows], args.seed)
    permuted_hybrid = hybrid.copy()
    permuted_hybrid[test] = permuted[test]

    group_check = []
    for level in sorted(np.unique(labels[test_rows])):
        members = test_rows[labels[test_rows] == level]
        same_marginals = bool(np.array_equal(np.sort(recorded[members], axis=0), np.sort(permuted[members], axis=0)))
        group_check.append({"cell_type": level, "n_cells": int(len(members)), "same_gene_distributions": same_marginals})
    if not all(item["same_gene_distributions"] for item in group_check):
        raise AssertionError("the permutation changed a gene distribution within a cell type")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_counts(hybrid_adata, permuted_hybrid, output / "hybrid.h5ad")
    write_counts(recorded_adata, permuted, output / "recorded.h5ad")
    for name in ("coordinates.parquet", "empty_coordinates.parquet", "splits.parquet"):
        shutil.copyfile(source / name, output / name)

    manifest = {
        "design": "test cells: recorded counts with every gene permuted independently within each cell type; fitting cells: the deployment hybrid unchanged",
        "deployment_dir": str(source),
        "deployment_hybrid_sha256": sha256_file(source / "hybrid.h5ad"),
        "deployment_recorded_sha256": sha256_file(source / "recorded.h5ad"),
        "label_column": args.label_column,
        "seed": args.seed,
        "n_cells": int(len(cell_ids)),
        "n_genes": int(recorded.shape[1]),
        "n_test_cells": int(test.sum()),
        "n_changed_test_entries": int(np.sum(permuted[test] != recorded[test])),
        "test_zero_fraction": float(np.mean(permuted[test] == 0)),
        "fitting_rows_identical": bool(np.array_equal(permuted_hybrid[~test], hybrid[~test])),
        "groups": group_check,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "groups"}, indent=2))

if __name__ == "__main__":
    main()
