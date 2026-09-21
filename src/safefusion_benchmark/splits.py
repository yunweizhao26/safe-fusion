from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


def split_biological_units(
    obs: pd.DataFrame,
    unit_column: str,
    development_fraction: float,
    validation_fraction: float,
    seed: int,
) -> pd.DataFrame:
    units = sorted(obs[unit_column].astype(str).unique())
    if len(units) < 3:
        raise ValueError("At least three biological units are required for development/validation/test splits")
    rng = np.random.default_rng(seed)
    units = list(np.asarray(units)[rng.permutation(len(units))])
    n_dev = max(1, int(np.floor(len(units) * development_fraction)))
    n_val = max(1, int(np.floor(len(units) * validation_fraction)))
    if n_dev + n_val >= len(units):
        n_dev, n_val = max(1, len(units) - 2), 1
    assignment = {unit: "development" for unit in units[:n_dev]}
    assignment.update({unit: "validation" for unit in units[n_dev:n_dev + n_val]})
    assignment.update({unit: "test" for unit in units[n_dev + n_val:]})
    result = pd.DataFrame({
        "cell_id": obs.index.astype(str),
        "biological_unit": obs[unit_column].astype(str).to_numpy(),
    })
    result["split"] = result["biological_unit"].map(assignment)
    if result.groupby("biological_unit")["split"].nunique().max() != 1:
        raise AssertionError("biological-unit leakage detected")
    return result
