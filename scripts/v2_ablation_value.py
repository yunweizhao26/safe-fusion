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
from sklearn.ensemble import HistGradientBoostingRegressor

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from run_leakage_safe_method import MAX_VALUE_FIT_ENTRIES, value_features
from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file
from v2_ablation_common import PRODUCTION_BOOSTING

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--teacher-contract", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-leaf-nodes", type=int, default=PRODUCTION_BOOSTING[0])
    parser.add_argument("--min-samples-leaf", type=int, default=PRODUCTION_BOOSTING[1])
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    training = np.isin(split, ["development", "validation"])

    teachers, own_counts_excluded = {}, []
    for path in args.teacher_contract:
        metadata = json.loads((Path(path) / "metadata.json").read_text())
        if metadata.get("scale") != "counts" or metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
            raise ValueError(f"teacher {path} is not a count-scale contract of this input")
        own_counts_excluded.append(bool(metadata["parameters"].get("fitting_cell_proposals_exclude_own_counts")))
        teachers[Path(path).name] = np.maximum(np.load(Path(path) / "mean.npy"), 0.0).astype(np.float32)
    names = list(teachers)

    coordinates = pd.read_parquet(args.coordinates)
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    keep = training[rows]
    rows, cols = rows[keep], cols[keep]
    if np.any(counts[rows, cols] != 0):
        raise ValueError("masked coordinates are not zero in the input counts")
    target = np.log1p(coordinates["original_value"].to_numpy(dtype=np.float64)[keep])
    fit_counts = counts[training]
    gene_mean = np.log1p(fit_counts.mean(axis=0)).astype(np.float32)
    detection = (fit_counts > 0).mean(axis=0).astype(np.float32)
    library_log = np.log1p(counts.sum(axis=1)).astype(np.float32)
    teacher_logs = np.column_stack([np.log1p(teachers[name][rows, cols]) for name in names]).astype(np.float32)
    features = value_features(teacher_logs, gene_mean, detection, library_log, rows, cols)

    rng = np.random.default_rng(args.seed)
    fit = np.sort(rng.choice(len(target), size=min(len(target), MAX_VALUE_FIT_ENTRIES), replace=False))
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf, early_stopping=True, validation_fraction=0.1, random_state=args.seed,
    )
    model.fit(features[fit], target[fit])

    prediction = np.empty(counts.shape, dtype=np.float32)
    all_cols = np.arange(counts.shape[1])
    for start in range(0, len(counts), 256):
        block = np.arange(start, min(start + 256, len(counts)))
        block_rows = np.repeat(block, len(all_cols))
        block_cols = np.tile(all_cols, len(block))
        block_logs = np.column_stack([np.log1p(teachers[name][block].ravel()) for name in names]).astype(np.float32)
        block_features = value_features(block_logs, gene_mean, detection, library_log, block_rows, block_cols)
        prediction[block] = model.predict(block_features).reshape(len(block), -1)
    prediction = np.expm1(np.maximum(prediction, 0.0)).astype(np.float32)

    metadata = {
        "method": "safe_fusion",
        "method_version": "v2_ablation_value",
        "scale": "counts",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["development", "validation"],
        "training_data": {"cell_ids_sha256": order_hash(np.asarray(cell_ids)[training].tolist())},
        "parameters": {
            "value_model": "boosted",
            "max_leaf_nodes": args.max_leaf_nodes,
            "min_samples_leaf": args.min_samples_leaf,
            "boosting_iterations": int(model.n_iter_),
            "teacher_names": names,
            "teacher_contracts": [str(path) for path in args.teacher_contract],
            "value_fit_entries": int(len(rows)),
            "value_fit_cells": "masked positives of model-fitting cells",
            "fit_cells": int(training.sum()),
            "heldout_test_cells": int((split == "test").sum()),
            "test_used_for_fit": False,
            "fitting_cell_proposals_exclude_own_counts": all(own_counts_excluded),
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.input),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, prediction, metadata)
    print(json.dumps({"output": args.output, "parameters": {k: v for k, v in metadata["parameters"].items() if k != "teacher_contracts"}}))

if __name__ == "__main__":
    main()
