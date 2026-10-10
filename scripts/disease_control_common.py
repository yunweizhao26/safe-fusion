#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"
DEPLOYMENT = EVIDENCE / "downstream_deployment"
OUTPUT = EVIDENCE / "disease_control_checks"
FRACTIONS = (0.01, 0.05, 0.10)
METHODS = ("Safe Fusion", "SVD", "Weighted kNN", "MAGIC", "scVI")
DRAWS = 2000
SEED = 1729

TISSUES = {
    "pancreas": {
        "units": ("pancreas_0", "pancreas_1", "pancreas_2"),
        "condition_column": "condition",
        "control": "Control",
        "cases": ("AAB", "T1D"),
        "masked_dataset": "Pancreas",
        "source": REPOSITORY / "external_data" / "cellxgene" / "f89a618b-fe4b-404e-bd39-7c574529b1f5.h5ad",
    },
    "colon": {
        "units": ("colon",),
        "condition_column": "disease",
        "control": "normal",
        "cases": ("Crohn disease",),
        "masked_dataset": "Colon",
        "source": REPOSITORY / "external_data" / "cellxgene" / "63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad",
    },
}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def unit_directory(key: str) -> Path:
    if key.startswith("pancreas_"):
        return DEPLOYMENT / "pancreas" / f"fold_{key.split('_')[1]}"
    return DEPLOYMENT / key


def fraction_suffix(fraction: float) -> str:
    return f"{fraction:g}".replace(".", "p")


def fill_contract(key: str, method: str, fraction: float) -> Path:
    percent = int(round(100 * fraction))
    unit = unit_directory(key)
    local = {"Safe Fusion": "safe_fusion", "SVD": "svd", "Weighted kNN": "weighted_knn"}
    if method in local:
        return unit / f"{local[method]}_{percent}pct"
    name = {"MAGIC": "magic", "scVI": "scvi"}[method]
    return OUTPUT / "deployment_fills" / key / f"{name}_topk_{fraction_suffix(fraction)}"


@dataclass
class HeldOut:
    tissue: str
    recorded: np.ndarray
    obs: pd.DataFrame
    gene_ids: list[str]
    symbols: np.ndarray
    test_masks: dict[str, np.ndarray]

    @property
    def heldout(self) -> np.ndarray:
        return self.obs["heldout"].to_numpy()

    @property
    def counts(self) -> np.ndarray:
        return self.recorded[self.heldout]

    @property
    def cells(self) -> pd.DataFrame:
        return self.obs.loc[self.heldout].reset_index(drop=True)


def donor_sex(tissue: str) -> dict[str, str]:
    source = ad.read_h5ad(TISSUES[tissue]["source"], backed="r")
    frame = source.obs[["donor_id", "sex"]].astype(str).drop_duplicates()
    if frame["donor_id"].duplicated().any():
        raise ValueError("a donor has more than one recorded sex")
    return dict(zip(frame["donor_id"], frame["sex"]))


def load_heldout(tissue: str) -> HeldOut:
    spec = TISSUES[tissue]
    first = unit_directory(spec["units"][0])
    adata = ad.read_h5ad(first / "recorded.h5ad")
    recorded = dense(adata.layers["corrupted_counts"]).astype(np.float32)
    cell_ids = adata.obs_names.astype(str)
    obs = adata.obs[["donor", "cell_type", "cell_type_broad", spec["condition_column"]]].astype(str).copy()
    if "sample_type" in adata.obs:
        obs["sample_type"] = adata.obs["sample_type"].astype(str)
    obs = obs.rename(columns={spec["condition_column"]: "condition_label"}).reset_index(names="cell_id")
    obs["fold"] = ""
    test_masks: dict[str, np.ndarray] = {}
    for key in spec["units"]:
        split = pd.read_parquet(unit_directory(key) / "splits.parquet").set_index("cell_id").loc[cell_ids, "split"]
        test = split.to_numpy() == "test"
        if (obs.loc[test, "fold"] != "").any():
            raise ValueError("a cell is held out in more than one unit")
        obs.loc[test, "fold"] = key
        test_masks[key] = test
    obs["heldout"] = obs["fold"] != ""
    sex = donor_sex(tissue)
    obs["sex"] = obs["donor"].map(sex)
    return HeldOut(
        tissue=tissue,
        recorded=recorded,
        obs=obs,
        gene_ids=adata.var_names.astype(str).tolist(),
        symbols=adata.var["feature_name"].astype(str).to_numpy(),
        test_masks=test_masks,
    )


def filled_matrix(data: HeldOut, method: str, fraction: float) -> np.ndarray:
    result = data.recorded.copy()
    for key, test in data.test_masks.items():
        contract = fill_contract(key, method, fraction)
        metadata = json.loads((contract / "metadata.json").read_text())
        if metadata["gene_ids"] != data.gene_ids or metadata["cell_ids"] != data.obs["cell_id"].tolist():
            raise ValueError(f"cell or gene order differs for {contract}")
        values = np.load(contract / "mean.npy", mmap_mode="r")
        filled = np.asarray(values[test], dtype=np.float32)
        recorded = data.recorded[test]
        if not np.array_equal(filled[recorded > 0], recorded[recorded > 0]):
            raise ValueError(f"{contract} changes recorded nonzero counts")
        result[test] = filled
    return result


def filled_mask(data: HeldOut, matrix: np.ndarray) -> np.ndarray:
    counts = data.counts
    return (counts == 0) & (matrix[data.heldout] != counts)


def stratified_draws(groups: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    levels = sorted(np.unique(groups))
    positions = [np.flatnonzero(groups == level) for level in levels]
    return np.stack([
        np.concatenate([rng.choice(members, size=len(members), replace=True) for members in positions])
        for _ in range(draws)
    ])


def interval(samples: np.ndarray) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return float("nan"), float("nan")
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def output_dir(analysis: str, tissue: str) -> Path:
    path = OUTPUT / analysis / tissue
    path.mkdir(parents=True, exist_ok=True)
    return path
