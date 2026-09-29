#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


CURATED_MARKERS = (
    "INS", "IAPP", "PCSK1", "PCSK2", "MAFA", "NKX6-1", "PDX1", "GCG", "TTR",
    "SST", "PPY", "GHRL", "PRSS1", "PRSS2", "REG1A", "REG1B", "CPA1", "CTRB2",
    "KRT8", "KRT18", "KRT19", "MUC1", "KRT17", "KRT7", "COL1A1", "COL1A2",
    "COL3A1", "SPARC", "DCN", "KDR", "EMCN", "PLVAP", "VWF", "PTPRC", "CD74",
    "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "CXCL10", "STAT1", "B2M", "IFITM1", "IFITM3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assignment(donors: pd.Series, seed: int) -> dict[str, str]:
    units = np.asarray(sorted(donors.astype(str).unique()))
    units = units[np.random.default_rng(seed).permutation(len(units))]
    n_dev = int(np.floor(0.50 * len(units)))
    n_val = int(np.floor(0.25 * len(units)))
    result = {unit: "development" for unit in units[:n_dev]}
    result.update({unit: "validation" for unit in units[n_dev:n_dev + n_val]})
    result.update({unit: "test" for unit in units[n_dev + n_val:]})
    return result


def stratified_cells(obs: pd.DataFrame, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, positions in obs.reset_index(drop=True).groupby(["donor_id", "cell_label", "disease_state"], observed=True, sort=True).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        if len(positions) > maximum:
            positions = rng.choice(positions, size=maximum, replace=False)
        keep.extend(positions.tolist())
    return np.asarray(sorted(keep), dtype=int)


def select_features(matrix: sparse.csr_matrix, development: np.ndarray, var: pd.DataFrame, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    fit = matrix[development].astype(np.float64)
    mean = np.asarray(fit.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(fit.power(2).mean(axis=0)).ravel() - mean**2, 0)
    detected = np.asarray((fit > 0).sum(axis=0)).ravel()
    score = np.divide(variance - mean, mean + 1e-8)
    score[detected < max(5, int(0.005 * fit.shape[0]))] = -np.inf
    ranked = np.argsort(-score, kind="stable")
    selected = ranked[np.isfinite(score[ranked])][:n_features]
    names = var["feature_name"].astype(str).to_numpy()
    curated = np.flatnonzero(np.isin(names, CURATED_MARKERS))
    selected = np.unique(np.concatenate([selected, curated])).astype(int)
    return selected, score[selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cells-per-stratum", type=int, default=12)
    parser.add_argument("--variable-genes", type=int, default=1200)
    parser.add_argument(
        "--gene-selection-splits",
        default=None,
        help="Splits parquet (cell_id, split) of one donor fold. The variable genes are then "
        "selected on its development cells, and locked_split records that fold's split. "
        "Default: the development donors of the 50/25/25 donor assignment of --seed.",
    )
    args = parser.parse_args()

    source_path = Path(args.input).resolve()
    source = ad.read_h5ad(source_path, backed="r")
    required = {"donor_id", "cell_label", "cell_type", "disease_state", "disease", "tissue"}
    missing = required - set(source.obs)
    if missing or source.raw is None:
        raise ValueError(f"missing metadata {sorted(missing)} or raw counts")
    donor_split = assignment(source.obs["donor_id"], args.seed)
    cells = stratified_cells(source.obs, args.cells_per_stratum, args.seed)
    obs = source.obs.iloc[cells].copy()
    counts = sparse.csr_matrix(source.raw.X[cells]).astype(np.int32)
    if np.any(counts.data < 0) or np.any(counts.data != np.floor(counts.data)):
        raise ValueError("source raw matrix is not non-negative integer counts")
    split = obs["donor_id"].astype(str).map(donor_split).to_numpy()
    selected_on = "development donors only"
    if args.gene_selection_splits:
        fold = pd.read_parquet(args.gene_selection_splits).set_index("cell_id")["split"]
        split = fold.reindex(obs.index.astype(str)).to_numpy()
        if pd.isna(split).any():
            raise ValueError("gene-selection splits do not cover every prepared cell")
        selected_on = f"development cells of {args.gene_selection_splits}"
    genes, score = select_features(counts, split == "development", source.raw.var, args.variable_genes)
    counts = counts[:, genes].tocsr()
    var = source.raw.var.iloc[genes].copy()
    var["selection_score"] = score
    var["curated_marker"] = var["feature_name"].astype(str).isin(CURATED_MARKERS).to_numpy()

    prepared_obs = pd.DataFrame(index=obs.index.astype(str))
    prepared_obs["donor"] = obs["donor_id"].astype(str).to_numpy()
    prepared_obs["cell_type"] = obs["cell_label"].astype(str).to_numpy()
    prepared_obs["cell_type_broad"] = obs["cell_type"].astype(str).to_numpy()
    prepared_obs["condition"] = obs["disease_state"].astype(str).to_numpy()
    prepared_obs["disease"] = obs["disease"].astype(str).to_numpy()
    prepared_obs["tissue"] = obs["tissue"].astype(str).to_numpy()
    prepared_obs["locked_split"] = split
    prepared = ad.AnnData(X=counts, obs=prepared_obs, var=var)
    prepared.layers["counts"] = counts.copy()
    prepared.uns["source"] = {
        "artifact": str(source_path), "sha256": sha256_file(source_path),
        "publication_doi": "10.1038/s42255-022-00531-x",
        "cellxgene_artifact_id": "f89a618b-fe4b-404e-bd39-7c574529b1f5",
    }
    prepared.uns["benchmark_preparation"] = {
        "seed": args.seed, "cells_per_donor_celltype_condition_stratum": args.cells_per_stratum,
        "variable_genes_selected_on": selected_on, "curated_markers_appended": list(CURATED_MARKERS),
        "donor_assignment": donor_split,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output, compression="gzip")
    report = {
        "source": str(source_path), "source_sha256": prepared.uns["source"]["sha256"],
        "output": str(output), "output_sha256": sha256_file(output),
        "n_cells": prepared.n_obs, "n_genes": prepared.n_vars,
        "n_donors": int(prepared.obs["donor"].nunique()), "n_cell_types": int(prepared.obs["cell_type"].nunique()),
        "cells_by_split": prepared.obs["locked_split"].value_counts().sort_index().to_dict(),
        "donors_by_split": prepared.obs.groupby("locked_split", observed=True)["donor"].nunique().sort_index().to_dict(),
        "conditions": prepared.obs["condition"].value_counts().sort_index().to_dict(),
        "raw_integer_counts": bool(np.all(counts.data == np.floor(counts.data))),
        "curated_markers_present": sorted(var.loc[var["curated_marker"], "feature_name"].astype(str).tolist()),
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
