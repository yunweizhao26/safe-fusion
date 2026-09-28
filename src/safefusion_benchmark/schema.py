from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from .io import counts_layer, dense


class SchemaError(ValueError):
    pass


def validate_dataset(adata: ad.AnnData, biological_unit: str) -> dict[str, Any]:
    errors: list[str] = []
    if not adata.obs_names.is_unique:
        errors.append("cell identifiers are not unique")
    if not adata.var_names.is_unique:
        errors.append("gene identifiers are not unique")
    if biological_unit not in adata.obs:
        errors.append(f"missing biological-unit field: {biological_unit}")
    counts = counts_layer(adata)
    values = counts.data if sparse.issparse(counts) else np.asarray(counts).ravel()
    if np.any(~np.isfinite(values)):
        errors.append("counts contain non-finite values")
    if np.any(values < 0):
        errors.append("counts contain negative values")
    if np.any(values != np.floor(values)):
        errors.append("raw counts are not integers")
    if not sparse.issparse(counts):
        errors.append("raw counts must be sparse")
    if errors:
        raise SchemaError("; ".join(errors))
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_biological_units": int(adata.obs[biological_unit].astype(str).nunique()),
        "counts_nnz": int(counts.nnz),
        "valid": True,
    }


def validate_protein_keys(panel: pd.DataFrame) -> dict[str, int]:
    required = {"cell_id", "reagent_id", "gene_id"}
    missing = required - set(panel.columns)
    if missing:
        raise SchemaError(f"protein panel missing columns: {sorted(missing)}")
    if panel[list(required)].isna().any().any():
        raise SchemaError("protein proxy keys contain missing values")
    duplicates = int(panel.duplicated(["cell_id", "reagent_id", "gene_id"]).sum())
    if duplicates:
        raise SchemaError(f"duplicate protein proxy keys: {duplicates}")
    return {"rows": len(panel), "distinct_reagents": panel["reagent_id"].nunique()}


def validate_perturbation_obs(obs: pd.DataFrame) -> None:
    required = {"guide_id", "guide_target", "condition", "replicate", "mixscape_class"}
    missing = required - set(obs.columns)
    if missing:
        raise SchemaError(f"perturbation metadata missing columns: {sorted(missing)}")
