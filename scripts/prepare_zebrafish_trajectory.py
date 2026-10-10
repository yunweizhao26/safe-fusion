#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def variable_genes(counts: sparse.csr_matrix, maximum: int) -> np.ndarray:
    library = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(10_000.0, library, out=np.zeros_like(library), where=library > 0)
    normalized = counts.multiply(scale[:, None]).tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - np.square(mean), 0.0)
    expressed = np.flatnonzero(np.asarray(counts.sum(axis=0)).ravel() > 0)
    return expressed[np.argsort(variance[expressed])[-maximum:]].astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-genes", type=int, default=2048)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = Path(args.input)
    adata = ad.read_h5ad(source)
    counts = adata.X.tocsr().astype(np.float32) if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X, dtype=np.float32)
    stages = adata.obs["Stage"].astype(str).to_numpy()
    ordered_stages = sorted(set(stages), key=lambda value: float(re.match(r"[0-9.]+", value).group()))
    stage_rank = {stage: index for index, stage in enumerate(ordered_stages)}
    rng = np.random.default_rng(args.seed)
    split = np.empty(adata.n_obs, dtype=object)
    for stage in ordered_stages:
        positions = rng.permutation(np.flatnonzero(stages == stage))
        n_test = max(1, int(round(args.test_fraction * len(positions))))
        split[positions[:n_test]] = "test"
        split[positions[n_test:]] = "development"
    development = np.flatnonzero(split == "development")
    genes = variable_genes(counts[development], args.max_genes)

    obs = adata.obs.copy()
    obs["condition"] = stages
    obs["target"] = "none"
    obs["control"] = 0
    obs["stage_rank"] = np.asarray([stage_rank[value] for value in stages], dtype=np.int16)
    obs["preassigned_split"] = split.astype(str)
    obs["source_cell_id"] = adata.obs_names.astype(str)
    obs.index = pd.Index([f"zebrafish_{index:05d}" for index in range(adata.n_obs)], name="cell_id")
    result_counts = counts[:, genes].tocsr()
    result = ad.AnnData(X=result_counts, obs=obs, var=adata.var.iloc[genes].copy())
    result.var_names = adata.var_names[genes]
    result.layers["counts"] = result_counts.copy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination, compression="gzip")
    report = {
        "source": str(source),
        "source_sha256": sha256(source),
        "dataset": "CellRank zebrafish axial mesoderm",
        "cells": int(result.n_obs),
        "genes": int(result.n_vars),
        "stages": ordered_stages,
        "n_stages": len(ordered_stages),
        "lineages": sorted(obs["lineages"].astype(str).unique()),
        "split_counts": pd.Series(split).value_counts().to_dict(),
        "design": "stage-stratified split first; variable genes selected on development cells only",
        "replicate_limitation": "processed object has no embryo or library replicate identifier",
        "seed": args.seed,
    }
    destination.with_suffix(".report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
