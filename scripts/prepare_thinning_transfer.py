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

from safefusion_benchmark.corruption import corrupt_counts
from safefusion_benchmark.hashing import sha256_file

MASK_SPEC = {"kind": "stratified_nonzero_mask", "fraction": 0.10, "gene_bins": 4, "library_bins": 4}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def split_of(path: str, cell_ids: np.ndarray) -> np.ndarray:
    return pd.read_parquet(path).set_index("cell_id").loc[cell_ids, "split"].astype(str).to_numpy()


def write_counts(obs: pd.DataFrame, var: pd.DataFrame, counts: np.ndarray, path: Path, uns: dict) -> None:
    matrix = sparse.csr_matrix(counts.astype(np.float32))
    result = ad.AnnData(X=matrix, obs=obs.copy(), var=var.copy())
    result.layers["corrupted_counts"] = matrix.copy()
    result.uns["corruption"] = json.dumps(uns)
    result.write_h5ad(path)


def thin(args: argparse.Namespace) -> None:
    truth = ad.read_h5ad(args.truth)
    counts = dense(truth.layers["counts"])
    if not np.array_equal(counts, np.round(counts)):
        raise ValueError("truth counts are not integers")
    counts = counts.astype(np.int64)
    cell_ids = truth.obs_names.astype(str).to_numpy()
    gene_ids = truth.var_names.astype(str).to_numpy()
    units = truth.obs[args.unit_column].astype(str).to_numpy()
    splits = split_of(args.splits, cell_ids)
    spec = {"kind": "binomial_thinning", "retained_fraction": float(args.retained_fraction)}
    thinned, _ = corrupt_counts(counts, cell_ids, gene_ids, units, splits, spec, args.seed)
    thinned = thinned.astype(np.int64)
    rows, cols = np.where((counts > 0) & (thinned == 0))
    coordinates = pd.DataFrame({
        "cell_id": cell_ids[rows],
        "gene_id": gene_ids[cols],
        "cell_index": rows.astype(np.int64),
        "gene_index": cols.astype(np.int64),
        "corruption_type": "binomial_thinning_zeroed",
        "original_value": counts[rows, cols].astype(np.float32),
        "corrupted_value": np.zeros(len(rows), dtype=np.float32),
        "biological_unit": units[rows],
        "split": splits[rows],
    })
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corruption = {**spec, "seed": args.seed}
    write_counts(truth.obs, truth.var, thinned, output / "corrupted.h5ad", corruption)
    coordinates.to_parquet(output / "coordinates.parquet", index=False)
    manifest = {
        "truth": str(args.truth),
        "truth_sha256": sha256_file(args.truth),
        "splits": str(args.splits),
        "corruption": corruption,
        "shape": list(counts.shape),
        "entries_set_to_zero": int(len(coordinates)),
        "original_count_1_share": float(np.mean(coordinates["original_value"] == 1)) if len(coordinates) else None,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


def hybrid(args: argparse.Namespace) -> None:
    thinned_dir = Path(args.thinned_dir)
    source = ad.read_h5ad(thinned_dir / "corrupted.h5ad")
    thinned = dense(source.layers["corrupted_counts"]).astype(np.int64)
    cell_ids = source.obs_names.astype(str).to_numpy()
    gene_ids = source.var_names.astype(str).to_numpy()
    split = split_of(args.splits, cell_ids)
    test = split == "test"
    if not test.any() or test.all():
        raise ValueError("the split file needs fitting and test cells")
    fitting = np.flatnonzero(~test)
    units = source.obs[args.unit_column].astype(str).to_numpy()

    masked_fitting, mask_coordinates = corrupt_counts(
        thinned[fitting], cell_ids[fitting], gene_ids, units[fitting], split[fitting], MASK_SPEC, args.mask_seed,
    )
    counts = thinned.copy()
    counts[fitting] = masked_fitting
    mask_coordinates["cell_index"] = fitting[mask_coordinates["cell_index"].to_numpy(dtype=np.int64)]

    thinning_coordinates = pd.read_parquet(thinned_dir / "coordinates.parquet")
    thinning_coordinates = thinning_coordinates[test[thinning_coordinates["cell_index"].to_numpy(dtype=np.int64)]].copy()
    thinning_coordinates["split"] = "test"
    coordinates = pd.concat([mask_coordinates, thinning_coordinates], ignore_index=True)
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    if np.any(counts[rows, cols] != 0) or coordinates.duplicated(["cell_index", "gene_index"]).any():
        raise AssertionError("coordinates must be unique zeros of the hybrid input")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    corruption = {
        "design": "fitting cells: thinned counts with the standard stratified mask; test cells: thinned counts",
        "thinned": str(thinned_dir / "corrupted.h5ad"),
        "mask": MASK_SPEC,
        "mask_seed": args.mask_seed,
    }
    write_counts(source.obs, source.var, counts, output / "hybrid.h5ad", corruption)
    coordinates.to_parquet(output / "coordinates.parquet", index=False)
    shutil.copyfile(args.splits, output / "splits.parquet")
    manifest = {
        **corruption,
        "thinned_sha256": sha256_file(thinned_dir / "corrupted.h5ad"),
        "splits": str(args.splits),
        "n_fitting_cells": int(len(fitting)),
        "n_test_cells": int(test.sum()),
        "n_fitting_masked_positives": int(len(mask_coordinates)),
        "fitting_masked_count_1_share": float(np.mean(mask_coordinates["original_value"] == 1)),
        "n_test_thinning_positives": int(len(thinning_coordinates)),
        "test_thinning_count_1_share": float(np.mean(thinning_coordinates["original_value"] == 1)),
        "n_test_zeros": int(np.sum(counts[test] == 0)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    thin_parser = commands.add_parser("thin", help="write a binomial-thinning benchmark input")
    thin_parser.add_argument("--truth", required=True, help="recorded counts in layers['counts']")
    thin_parser.add_argument("--splits", required=True)
    thin_parser.add_argument("--unit-column", required=True)
    thin_parser.add_argument("--retained-fraction", type=float, required=True)
    thin_parser.add_argument("--seed", type=int, default=1729)
    thin_parser.add_argument("--output-dir", required=True)
    hybrid_parser = commands.add_parser("hybrid", help="write the deployment-matched input of one split")
    hybrid_parser.add_argument("--thinned-dir", required=True, help="thinning benchmark input directory")
    hybrid_parser.add_argument("--splits", required=True)
    hybrid_parser.add_argument("--unit-column", required=True)
    hybrid_parser.add_argument("--mask-seed", type=int, default=1729)
    hybrid_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    thin(args) if args.command == "thin" else hybrid(args)


if __name__ == "__main__":
    main()
