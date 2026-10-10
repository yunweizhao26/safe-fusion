#!/usr/bin/env python3






from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = ROOT / "artifacts/paper_evidence/review_round4/protein_full"
sys.path.insert(0, str(ROOT / "scripts"))
sys.dont_write_bytecode = True

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import prepare_external_perturbseq as original


def main() -> None:
    extension = argparse.ArgumentParser(add_help=False)
    extension.add_argument("--matched-panel")
    extension.add_argument("--gene-panel")
    extended, _ = extension.parse_known_args()
    if extended.matched_panel is None and extended.gene_panel is None:
        original.main()
        return
    if not extended.matched_panel or not extended.gene_panel:
        extension.error("--matched-panel and --gene-panel must be supplied together")
    parser = argparse.ArgumentParser(description=__doc__, parents=[extension])
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True, choices=["papalexi_eccite"])
    parser.add_argument("--seed", type=int, default=1729, choices=[1729])
    args = parser.parse_args()
    started = time.monotonic()
    destination = Path(args.output).resolve()
    if not destination.is_relative_to(OUTPUT_ROOT):
        raise ValueError(f"Output must be under {OUTPUT_ROOT}")
    report_path = destination.with_suffix(".report.json")
    split_path = destination.with_suffix(".split_by_target.csv")
    for path in (destination, report_path, split_path):
        if path.exists():
            raise FileExistsError(path)

    source = Path(args.input).resolve()
    gene_panel_path = Path(args.gene_panel).resolve()
    panel_path = Path(args.matched_panel).resolve()
    raw = ad.read_h5ad(source, backed="r")
    frozen = ad.read_h5ad(gene_panel_path)
    panel = pd.read_parquet(panel_path)
    if not raw.obs_names.is_unique or not raw.var_names.is_unique:
        raise ValueError("Raw cell and gene identifiers must be unique")
    if frozen.shape != (3400, 2032) or not frozen.var_names.is_unique:
        raise ValueError(f"Unexpected frozen prepared shape/genes: {frozen.shape}")
    if panel[["cell_id", "target", "guide_id"]].isna().any().any():
        raise ValueError("Audit panel contains missing identity fields")
    identities = panel[["cell_id", "target", "guide_id"]].astype(str).drop_duplicates()
    if identities.cell_id.duplicated().any() or len(identities) != 20156:
        raise ValueError("Audit must identify exactly 20,156 unambiguous matched cells")
    identities = identities.set_index("cell_id")
    missing_cells = identities.index.difference(raw.obs_names)
    if len(missing_cells):
        raise ValueError(f"Audit cells absent from raw RNA: {missing_cells.tolist()[:10]}")
    selected = np.flatnonzero(raw.obs_names.isin(identities.index))
    source_ids = raw.obs_names[selected].astype(str)
    gene_indices = raw.var_names.get_indexer(frozen.var_names)
    if np.any(gene_indices < 0):
        raise ValueError("Frozen genes are absent from raw RNA")
    all_targets, all_control = original.dataset_annotations(raw, args.dataset)
    targets, control = all_targets[selected], all_control[selected]
    audit_targets = identities.loc[source_ids, "target"].replace({"NT": "none"}).to_numpy()
    if not np.array_equal(targets, audit_targets):
        raise ValueError("Raw perturbation targets differ from audit targets")
    if not np.array_equal(
        raw.obs.iloc[selected]["guide_id"].astype(str).to_numpy(),
        identities.loc[source_ids, "guide_id"].to_numpy(),
    ):
        raise ValueError("Raw guide assignments differ from audit guides")
    split = original.stratified_split(targets, control, 0.30, np.random.default_rng(args.seed))
    if set(split) != {"development", "test"}:
        raise ValueError(f"Unexpected split labels: {set(split)}")
    split_counts = pd.crosstab(pd.Series(targets, name="target"), pd.Series(split, name="split"))
    expected_test = np.maximum(1, np.rint(split_counts.sum(axis=1).to_numpy() * 0.30)).astype(int)
    np.testing.assert_array_equal(split_counts["test"].to_numpy(), expected_test)

    print(f"[prepare-full] materializing {len(selected)} cells x {len(gene_indices)} frozen genes", flush=True)

    selected_counts = sparse.csr_matrix(raw.X[:, gene_indices], dtype=np.float32)[selected]
    frozen_ids = frozen.obs["source_cell_id"].astype(str)
    if not frozen_ids.is_unique or not frozen_ids.isin(raw.obs_names).all():
        raise ValueError("Frozen prepared source identifiers are invalid")
    overlap = frozen_ids.isin(source_ids).to_numpy()
    overlap_rows = source_ids.get_indexer(frozen_ids[overlap])
    frozen_counts = sparse.csr_matrix(frozen.X, dtype=np.float32)[overlap]
    if not overlap.any() or (selected_counts[overlap_rows] != frozen_counts).nnz:
        raise ValueError("Counts differ from frozen prepared data on overlapping cells")
    np.testing.assert_array_equal(targets[overlap_rows], frozen.obs.loc[overlap, "target"].astype(str))

    obs = raw.obs.iloc[selected].copy()
    obs["source_cell_id"] = source_ids.to_numpy()
    obs["condition"] = np.where(control, "ctrl", targets)
    obs["target"] = targets
    obs["control"] = control.astype(np.int8)
    obs["preassigned_split"] = split.astype(str)
    obs.index = pd.Index([f"{args.dataset}_{index:05d}" for index in range(len(obs))], name="cell_id")
    result = ad.AnnData(X=selected_counts, obs=obs, var=raw.var.iloc[gene_indices].copy())
    result.layers["counts"] = selected_counts.copy()
    np.testing.assert_array_equal(result.var_names, frozen.var_names)
    if result.shape != (20156, 2032):
        raise ValueError(f"Unexpected full benchmark shape: {result.shape}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination, compression="gzip")
    split_counts.to_csv(split_path)
    raw.file.close()
    report = {
        "source": str(source),
        "source_sha256": original.sha256(source),
        "matched_panel": str(panel_path),
        "matched_panel_sha256": original.sha256(panel_path),
        "gene_panel": str(gene_panel_path),
        "gene_panel_sha256": original.sha256(gene_panel_path),
        "original_preparation_sha256": original.sha256(Path(original.__file__)),
        "wrapper_sha256": original.sha256(Path(__file__)),
        "output": str(destination),
        "output_sha256": original.sha256(destination),
        "dataset": args.dataset,
        "design": "all audit-matched cells in raw order; frozen 2,032-gene panel; original target-stratified 30% test split",
        "seed": args.seed,
        "test_fraction": 0.30,
        "cells": result.n_obs,
        "genes": result.n_vars,
        "conditions": len(set(targets) - {"none"}),
        "condition_targets": sorted(set(targets) - {"none"}),
        "control_cells": int(control.sum()),
        "split_counts": pd.Series(split).value_counts().to_dict(),
        "split_by_target": split_counts.reset_index().to_dict(orient="records"),
        "overlapping_frozen_cells": int(overlap.sum()),
        "overlapping_counts_identical": True,
        "gene_order_identical": True,
        "audit_targets_and_guides_identical": True,
        "elapsed_seconds": time.monotonic() - started,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
