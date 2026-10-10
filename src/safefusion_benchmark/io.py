from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def counts_layer(adata: ad.AnnData) -> sparse.csr_matrix:
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    return sparse.csr_matrix(matrix)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def make_synthetic_counts(n_cells: int, n_genes: int, n_units: int, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), int(np.ceil(n_cells / n_units)))[:n_cells]
    cell_type = np.arange(n_cells) % 3
    condition = np.arange(n_cells) % 2
    pseudotime = np.linspace(0.0, 1.0, n_cells)
    lineage = (np.arange(n_cells) % 4 >= 2).astype(int)
    gene_base = rng.gamma(1.8, 1.2, n_genes)
    effects = rng.lognormal(0.0, 0.35, (3, n_genes))
    time_effect = np.ones((n_cells, n_genes))
    time_effect[:, : max(2, n_genes // 5)] *= np.exp(0.9 * pseudotime[:, None])
    condition_effect = np.ones((n_cells, n_genes))
    condition_effect[:, -max(2, n_genes // 6):] *= np.where(condition[:, None] == 1, 1.7, 1.0)
    means = gene_base[None, :] * effects[cell_type] * time_effect * condition_effect
    clean = rng.poisson(means).astype(np.int32)
    biological_zero = rng.random(clean.shape) < 0.12
    clean[biological_zero] = 0
    observed = rng.binomial(clean, 0.72).astype(np.int32)
    obs = pd.DataFrame(
        {
            "donor": [f"unit_{i}" for i in unit],
            "cell_type": [f"type_{i}" for i in cell_type],
            "condition": [f"condition_{i}" for i in condition],
            "pseudotime": pseudotime,
            "lineage": [f"branch_{i}" for i in lineage],
        },
        index=[f"cell_{i:04d}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i:04d}" for i in range(n_genes)])
    adata = ad.AnnData(X=sparse.csr_matrix(observed), obs=obs, var=var)
    adata.layers["counts"] = sparse.csr_matrix(observed)
    adata.layers["clean_truth"] = sparse.csr_matrix(clean)
    adata.layers["biological_zero"] = sparse.csr_matrix(biological_zero.astype(np.int8))
    adata.uns["generated"] = {"model": "sergio_like_smoke", "seed": seed}
    edges = [[f"gene_{i:04d}", f"gene_{i + 1:04d}"] for i in range(min(8, n_genes - 1))]
    adata.uns["grn_edges"] = edges
    return adata
