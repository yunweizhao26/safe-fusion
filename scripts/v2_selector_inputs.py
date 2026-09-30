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

from masked_f1_units import load_units_manifest
from safefusion_benchmark.hashing import sha256_file

DESIGNS = ("entry", "molecule", "entry_molecule")
MANIFEST = REPOSITORY / "artifacts/paper_evidence/review_round2/leakage_free/units_manifest.json"
INNER_FRACTION = 0.2
INNER_THINNING = 0.5
STREAM = {"inner_split": 0, "inner_thinning": 1, "design": 2}

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def read_counts(path: Path, layer: str) -> tuple[ad.AnnData, np.ndarray]:
    adata = ad.read_h5ad(path)
    counts = dense(adata.layers[layer] if layer != "X" else adata.X)
    if not np.array_equal(counts, np.round(counts)):
        raise ValueError(f"{path} layer {layer} is not integer")
    return adata, counts.astype(np.int64)

def split_of(path: Path, cell_ids: np.ndarray) -> np.ndarray:
    return pd.read_parquet(path).set_index("cell_id").loc[cell_ids, "split"].astype(str).to_numpy()

def positive_matrix(coordinates: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    matrix = np.zeros(shape, dtype=bool)
    matrix[coordinates["cell_index"].to_numpy(dtype=np.int64), coordinates["gene_index"].to_numpy(dtype=np.int64)] = True
    return matrix

def apply_design(design: str, rho: float, recorded: np.ndarray, masked: np.ndarray, entry_positive: np.ndarray,
                 seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if design == "entry":
        return masked.copy(), entry_positive.copy(), recorded
    rng = np.random.default_rng((seed, STREAM["design"]))
    source = recorded if design == "molecule" else masked
    thinned = rng.binomial(source, 1.0 - rho)
    positive = (source > 0) & (thinned == 0)
    if design == "entry_molecule":
        positive |= entry_positive
    return thinned.astype(np.int64), positive, recorded

def coordinate_frame(positive: np.ndarray, original: np.ndarray, rows: np.ndarray, cell_ids: np.ndarray,
                     gene_ids: np.ndarray, units: np.ndarray, split: np.ndarray, kind: str,
                     evaluation: str) -> pd.DataFrame:

    local, cols = np.where(positive)
    global_rows = rows[local]
    return pd.DataFrame({
        "cell_id": cell_ids[global_rows],
        "gene_id": gene_ids[cols],
        "cell_index": global_rows.astype(np.int64),
        "gene_index": cols.astype(np.int64),
        "corruption_type": kind,
        "original_value": original[local, cols].astype(np.float32),
        "corrupted_value": np.zeros(len(local), dtype=np.float32),
        "biological_unit": units[global_rows],
        "split": split[global_rows],
        "evaluation": evaluation,
    })

def write_input(output: Path, obs: pd.DataFrame, var: pd.DataFrame, counts: np.ndarray, coordinates: pd.DataFrame,
                splits: pd.DataFrame, manifest: dict) -> None:
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    if np.any(counts[rows, cols] != 0) or coordinates.duplicated(["cell_index", "gene_index"]).any():
        raise AssertionError("coordinates must be unique zeros of the input")
    output.mkdir(parents=True, exist_ok=True)
    matrix = sparse.csr_matrix(counts.astype(np.float32))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.layers["corrupted_counts"] = matrix.copy()
    adata.uns["corruption"] = json.dumps(manifest["design"])
    adata.write_h5ad(output / "input.h5ad")
    coordinates.to_parquet(output / "coordinates.parquet", index=False)
    splits.to_parquet(output / "splits.parquet", index=False)
    manifest["input_sha256"] = sha256_file(output / "input.h5ad")
    manifest["positives"] = coordinates.groupby("evaluation").size().to_dict()
    manifest["count_one_share"] = coordinates.groupby("evaluation")["original_value"].apply(
        lambda values: float(np.mean(values == 1))).to_dict()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps(manifest, default=str))

def inner_test_cells(unit_labels: np.ndarray, roles: np.ndarray, by_cell: bool, seed: int) -> np.ndarray:

    rng = np.random.default_rng((seed, STREAM["inner_split"]))
    chosen = np.zeros(len(unit_labels), dtype=bool)
    for role in sorted(set(roles)):
        in_role = roles == role
        if by_cell:
            for label in sorted(set(unit_labels[in_role])):
                members = np.flatnonzero(in_role & (unit_labels == label))
                n = int(round(INNER_FRACTION * len(members)))
                chosen[rng.choice(members, size=n, replace=False)] = True
        else:
            donors = np.asarray(sorted(set(unit_labels[in_role])))
            n = max(1, int(round(INNER_FRACTION * len(donors))))
            held = rng.choice(donors, size=n, replace=False)
            chosen |= in_role & np.isin(unit_labels, held)
    return chosen

def inner(args: argparse.Namespace) -> None:
    unit = {unit.key: unit for unit in load_units_manifest(args.units_manifest)}[args.unit]
    corrupted_adata, masked_all = read_counts(unit.corrupted, "corrupted_counts")
    truth_adata, recorded_all = read_counts(unit.truth, "counts")
    if not (np.array_equal(corrupted_adata.obs_names, truth_adata.obs_names)
            and np.array_equal(corrupted_adata.var_names, truth_adata.var_names)):
        raise ValueError("benchmark input and recorded counts differ in order")
    cell_ids = corrupted_adata.obs_names.to_numpy().astype(str)
    gene_ids = corrupted_adata.var_names.to_numpy().astype(str)
    split = split_of(unit.splits, cell_ids)
    training = np.flatnonzero(np.isin(split, ["development", "validation"]))
    labels = corrupted_adata.obs[unit.unit_column].astype(str).to_numpy()
    entry_positive_all = positive_matrix(pd.read_parquet(unit.coordinates), masked_all.shape)

    held = inner_test_cells(labels[training], split[training], unit.dataset == "CRISPRa", args.seed)
    fit_rows, test_rows = training[~held], training[held]
    rng = np.random.default_rng((args.seed, STREAM["inner_thinning"]))
    thinned_test = rng.binomial(recorded_all[test_rows], INNER_THINNING).astype(np.int64)
    thinned_positive = (recorded_all[test_rows] > 0) & (thinned_test == 0)

    fit_counts, fit_positive, fit_original = apply_design(
        args.design, args.rho, recorded_all[fit_rows], masked_all[fit_rows], entry_positive_all[fit_rows], args.seed)
    blocks = [
        ("fit", fit_rows, fit_counts, fit_positive, fit_original, split[fit_rows]),
        ("masked", test_rows, masked_all[test_rows], entry_positive_all[test_rows], recorded_all[test_rows],
         np.full(len(test_rows), "test")),
        ("thinned", test_rows, thinned_test, thinned_positive, recorded_all[test_rows], np.full(len(test_rows), "test")),
    ]
    obs_parts, split_parts = [], []
    for evaluation, rows, _, _, _, roles in blocks:
        block_ids = cell_ids[rows] if evaluation == "fit" else np.char.add(cell_ids[rows], f"|{evaluation}")
        obs = corrupted_adata.obs.iloc[rows].copy()
        obs.index = block_ids
        obs["source_cell_id"] = cell_ids[rows]
        obs["evaluation"] = evaluation
        obs_parts.append(obs)
        split_parts.append(pd.DataFrame({"cell_id": block_ids, "biological_unit": labels[rows], "split": roles,
                                         "evaluation": evaluation}))
    splits = pd.concat(split_parts, ignore_index=True)
    ids = splits["cell_id"].to_numpy()
    roles = splits["split"].to_numpy()
    units = splits["biological_unit"].to_numpy()
    starts = np.cumsum([0] + [len(block[1]) for block in blocks])
    coordinate_parts = []
    for index, (evaluation, _, _, positive, original, _) in enumerate(blocks):
        kind = {"fit": args.design, "masked": "stratified_nonzero_mask", "thinned": "binomial_thinning"}[evaluation]
        coordinate_parts.append(coordinate_frame(
            positive, original, np.arange(starts[index], starts[index + 1]), ids, gene_ids, units, roles, kind, evaluation))
    manifest = {
        "command": "inner",
        "unit": unit.key,
        "dataset": unit.dataset,
        "design": {"design": args.design, "rho": args.rho if args.design != "entry" else None, "seed": args.seed},
        "inner_split": {
            "fraction": INNER_FRACTION,
            "unit": "cells of each perturbation" if unit.dataset == "CRISPRa" else "training donors within each split role",
            "inner_test_units": sorted(set(labels[test_rows])) if unit.dataset != "CRISPRa" else None,
            "n_fit_cells": int(len(fit_rows)),
            "n_inner_test_cells": int(len(test_rows)),
            "fit_roles": pd.Series(split[fit_rows]).value_counts().to_dict(),
        },
        "inner_thinning_retained_fraction": INNER_THINNING,
        "source": {"corrupted": str(unit.corrupted), "truth": str(unit.truth), "splits": str(unit.splits),
                   "coordinates": str(unit.coordinates)},
    }
    write_input(args.output_dir, pd.concat(obs_parts), corrupted_adata.var.copy(),
                np.concatenate([block[2] for block in blocks]), pd.concat(coordinate_parts, ignore_index=True),
                splits, manifest)

def refit(args: argparse.Namespace) -> None:
    source, masked_all = read_counts(args.input, "corrupted_counts")
    recorded_adata, recorded_all = read_counts(args.recorded, args.recorded_layer)
    if not np.array_equal(source.var_names, recorded_adata.var_names):
        raise ValueError("input and recorded counts differ in genes")
    recorded_all = pd.DataFrame(recorded_all, index=recorded_adata.obs_names.astype(str)).loc[
        source.obs_names.astype(str)].to_numpy()
    cell_ids = source.obs_names.to_numpy().astype(str)
    gene_ids = source.var_names.to_numpy().astype(str)
    split = split_of(args.splits, cell_ids)
    coordinates = pd.read_parquet(args.coordinates)
    fitting = np.flatnonzero(split != "test")
    in_fitting = np.isin(coordinates["cell_index"].to_numpy(dtype=np.int64), fitting)
    entry_positive = positive_matrix(coordinates.loc[in_fitting], masked_all.shape)[fitting]
    unmasked = masked_all[fitting].copy()
    unmasked[entry_positive] = recorded_all[fitting][entry_positive]
    if not np.array_equal(unmasked, recorded_all[fitting]):
        raise ValueError("fitting cells of the input differ from the recorded counts outside the entry mask")
    fit_counts, fit_positive, fit_original = apply_design(
        args.design, args.rho, recorded_all[fitting], masked_all[fitting], entry_positive, args.seed)
    counts = masked_all.copy()
    counts[fitting] = fit_counts
    units = (source.obs[args.unit_column].astype(str).to_numpy() if args.unit_column
             else np.asarray(["" for _ in cell_ids]))
    fit_coordinates = coordinate_frame(fit_positive, fit_original, fitting, cell_ids, gene_ids, units, split,
                                       args.design, "fit")
    kept = coordinates.loc[~in_fitting].copy()
    kept["evaluation"] = "test"
    frame = pd.concat([fit_coordinates, kept], ignore_index=True)
    splits = pd.read_parquet(args.splits)
    manifest = {
        "command": "refit",
        "design": {"design": args.design, "rho": args.rho if args.design != "entry" else None, "seed": args.seed},
        "source": {"input": str(args.input), "coordinates": str(args.coordinates), "splits": str(args.splits),
                   "recorded": str(args.recorded), "recorded_layer": args.recorded_layer},
        "n_fitting_cells": int(len(fitting)),
        "n_test_cells": int((split == "test").sum()),
    }
    write_input(args.output_dir, source.obs.copy(), source.var.copy(), counts, frame, splits, manifest)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inner", "refit"):
        command = commands.add_parser(name)
        command.add_argument("--design", choices=DESIGNS, required=True)
        command.add_argument("--rho", type=float, default=0.0, help="molecule removal probability")
        command.add_argument("--seed", type=int, default=1729)
        command.add_argument("--output-dir", type=Path, required=True)
        if name == "inner":
            command.add_argument("--units-manifest", type=Path, default=MANIFEST)
            command.add_argument("--unit", required=True)
        else:
            command.add_argument("--input", type=Path, required=True, help="h5ad with layers['corrupted_counts']")
            command.add_argument("--coordinates", type=Path, required=True)
            command.add_argument("--splits", type=Path, required=True)
            command.add_argument("--recorded", type=Path, required=True, help="h5ad with the uncorrupted counts")
            command.add_argument("--recorded-layer", default="counts")
            command.add_argument("--unit-column", default=None)
    args = parser.parse_args()
    if args.design != "entry" and not 0.0 < args.rho < 1.0:
        parser.error("--rho must lie in (0, 1) for molecule designs")
    inner(args) if args.command == "inner" else refit(args)

if __name__ == "__main__":
    main()
