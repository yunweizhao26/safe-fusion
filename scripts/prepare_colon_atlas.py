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
    "BEST4", "OTOP2", "CA7", "GUCA2A", "GUCA2B", "SPIB", "CFTR",
    "MUC2", "TFF1", "TFF3", "AGR2", "SPDEF", "POU2F3", "DCLK1",
    "SPINK4", "CA1", "CA2", "TMIGD1", "MEP1A", "LGR5", "OLFM4",
    "MKI67", "PCNA", "CHGA", "GCG", "GIP", "CCK", "ASCL2", "SOX9",
    "LYZ", "DEFA5", "REG1A", "REG3A", "DUOX2", "NOS2", "CXCL1",
    "CXCL2", "CXCL3", "CXCL8", "CCL20", "IL32", "HLA-DRA", "HLA-DPA1",
    "HLA-DPB1", "HLA-A", "HLA-B", "STAT1", "IRF1", "IFITM1", "IFITM3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def donor_assignment(donors: pd.Series, seed: int) -> dict[str, str]:
    units = np.asarray(sorted(donors.astype(str).unique()))
    units = units[np.random.default_rng(seed).permutation(len(units))]
    n_dev = int(np.floor(0.50 * len(units)))
    n_val = int(np.floor(0.25 * len(units)))
    assignment = {unit: "development" for unit in units[:n_dev]}
    assignment.update({unit: "validation" for unit in units[n_dev:n_dev + n_val]})
    assignment.update({unit: "test" for unit in units[n_dev + n_val:]})
    return assignment


def stratified_cells(obs: pd.DataFrame, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    columns = ["donor_id", "Celltype", "Type"]
    for _, positions in obs.reset_index(drop=True).groupby(columns, observed=True, sort=True).groups.items():
        positions = np.asarray(list(positions), dtype=int)
        if len(positions) > maximum:
            positions = rng.choice(positions, size=maximum, replace=False)
        keep.extend(positions.tolist())
    return np.asarray(sorted(keep), dtype=int)


def feature_selection(
    matrix: sparse.csr_matrix,
    development: np.ndarray,
    var: pd.DataFrame,
    n_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit = matrix[development].astype(np.float64)
    mean = np.asarray(fit.mean(axis=0)).ravel()
    mean_sq = np.asarray(fit.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0.0)
    detected = np.asarray((fit > 0).sum(axis=0)).ravel()
    score = np.divide(variance - mean, mean + 1e-8)
    score[detected < max(5, int(0.005 * fit.shape[0]))] = -np.inf
    ranked = np.argsort(-score, kind="stable")
    selected = ranked[np.isfinite(score[ranked])][:n_features]

    feature_names = var["feature_name"].astype(str).to_numpy()
    curated = np.flatnonzero(np.isin(feature_names, CURATED_MARKERS))
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
    args = parser.parse_args()

    source_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    source = ad.read_h5ad(source_path, backed="r")
    required = {"donor_id", "biosample_id", "Type", "Celltype", "cell_type", "disease", "tissue"}
    missing = required - set(source.obs)
    if missing:
        raise ValueError(f"source is missing required metadata: {sorted(missing)}")
    if source.raw is None:
        raise ValueError("source .raw with integer UMI counts is required")

    assignment = donor_assignment(source.obs["donor_id"], args.seed)
    selected_cells = stratified_cells(source.obs, args.cells_per_stratum, args.seed)
    obs = source.obs.iloc[selected_cells].copy()
    raw = sparse.csr_matrix(source.raw.X[selected_cells]).astype(np.int32)
    if np.any(raw.data < 0) or np.any(raw.data != np.floor(raw.data)):
        raise ValueError("source .raw is not non-negative integer counts")
    split = obs["donor_id"].astype(str).map(assignment).to_numpy()
    selected_genes, selection_score = feature_selection(
        raw, split == "development", source.raw.var, args.variable_genes
    )
    raw = raw[:, selected_genes].tocsr()
    var = source.raw.var.iloc[selected_genes].copy()
    var["selection_score"] = selection_score
    var["curated_marker"] = var["feature_name"].astype(str).isin(CURATED_MARKERS).to_numpy()

    prepared_obs = pd.DataFrame(index=obs.index.astype(str))
    prepared_obs["donor"] = obs["donor_id"].astype(str).to_numpy()
    prepared_obs["biosample"] = obs["biosample_id"].astype(str).to_numpy()
    prepared_obs["cell_type"] = obs["Celltype"].astype(str).to_numpy()
    prepared_obs["cell_type_broad"] = obs["cell_type"].astype(str).to_numpy()
    prepared_obs["sample_type"] = obs["Type"].astype(str).to_numpy()
    prepared_obs["condition"] = np.where(obs["Type"].astype(str).eq("Infl"), "inflamed", "not_inflamed")
    prepared_obs["disease"] = obs["disease"].astype(str).to_numpy()
    prepared_obs["tissue"] = obs["tissue"].astype(str).to_numpy()
    prepared_obs["locked_split"] = split

    prepared = ad.AnnData(X=raw, obs=prepared_obs, var=var)
    prepared.layers["counts"] = raw.copy()
    prepared.uns["source"] = {
        "artifact": str(source_path),
        "sha256": sha256_file(source_path),
        "publication_doi": "10.1016/j.immuni.2023.01.002",
        "cellxgene_artifact_id": "63ff2c52-cb63-44f0-bac3-d0b33373e312",
    }
    prepared.uns["benchmark_preparation"] = {
        "seed": args.seed,
        "cells_per_donor_celltype_sampletype_stratum": args.cells_per_stratum,
        "variable_genes_selected_on": "development donors only",
        "n_variable_genes_requested": args.variable_genes,
        "curated_markers_appended": list(CURATED_MARKERS),
        "donor_assignment": assignment,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output_path, compression="gzip")

    counts_by_split = prepared.obs["locked_split"].value_counts().sort_index().to_dict()
    donors_by_split = prepared.obs.groupby("locked_split", observed=True)["donor"].nunique().sort_index().to_dict()
    report = {
        "source": str(source_path),
        "source_sha256": prepared.uns["source"]["sha256"],
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "n_cells": int(prepared.n_obs),
        "n_genes": int(prepared.n_vars),
        "n_donors": int(prepared.obs["donor"].nunique()),
        "n_biosamples": int(prepared.obs["biosample"].nunique()),
        "n_cell_types": int(prepared.obs["cell_type"].nunique()),
        "cells_by_split": {str(k): int(v) for k, v in counts_by_split.items()},
        "donors_by_split": {str(k): int(v) for k, v in donors_by_split.items()},
        "counts_nnz": int(raw.nnz),
        "counts_integer": bool(np.all(raw.data == np.floor(raw.data))),
        "curated_markers_present": sorted(var.loc[var["curated_marker"], "feature_name"].astype(str).tolist()),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
