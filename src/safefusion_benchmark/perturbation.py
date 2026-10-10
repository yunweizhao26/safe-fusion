from __future__ import annotations

import pandas as pd

from .schema import SchemaError, validate_perturbation_obs


def audit_eccite_library(
    obs: pd.DataFrame,
    expected_guides: int = 111,
    expected_targets: int = 26,
    expected_nt_guides: int = 10,
) -> dict[str, int]:
    validate_perturbation_obs(obs)
    guide_table = obs[["guide_id", "guide_target"]].drop_duplicates()
    if guide_table["guide_id"].duplicated().any():
        raise SchemaError("a guide ID maps to more than one target")
    targets = set(guide_table.loc[~guide_table["guide_target"].isin(["NT", "control"]), "guide_target"])
    nt_guides = guide_table[guide_table["guide_target"].isin(["NT", "control"])]["guide_id"].nunique()
    observed = {"guides": len(guide_table), "targets": len(targets), "nt_guides": int(nt_guides)}
    expected = {"guides": expected_guides, "targets": expected_targets, "nt_guides": expected_nt_guides}
    if observed != expected:
        raise SchemaError(f"ECCITE library mismatch: observed={observed}, expected={expected}")
    return observed


def effective_perturbation_mask(obs: pd.DataFrame) -> pd.Series:
    validate_perturbation_obs(obs)
    return obs["mixscape_class"].astype(str).eq("KO")


def cross_modal_mask(obs: pd.DataFrame, protein_to_gene: dict[str, str], readout: str) -> pd.Series:
    validate_perturbation_obs(obs)
    if readout not in protein_to_gene:
        raise KeyError(readout)
    return effective_perturbation_mask(obs) & obs["guide_target"].astype(str).ne(protein_to_gene[readout])
