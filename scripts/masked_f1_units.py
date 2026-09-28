#!/usr/bin/env python3
"""Evaluation units, comparator contracts and ranking scale for the masked-F1 comparison.

Every comparator ranks the candidate zeros of the held-out cells by its own
value on the count scale of the masked input. A contract's metadata ``scale``
decides the conversion, so depth-normalized outputs regain the sequencing depth
that the count-scale methods (and the Safe Fusion selector) see.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "paper_evidence"
PANCREAS_RUN = ROOT / "artifacts" / "pancreas_runs" / "0b2469810675-45c81b160d78"
COLON_RUN = ROOT / "artifacts" / "colon_runs" / "0b2469810675-c0db6f963e94"
COLON_METHODS = COLON_RUN / "methods" / "standardized" / "colon_epithelial" / "mask_010"

DATASETS = ("Pancreas", "Colon", "CRISPRa")
CURVE_BUDGETS = np.linspace(0.001, 1.0, 1000)
# Fill fractions at which unit-level counts are kept for paired intervals.
UNIT_FRACTIONS = tuple(round(0.01 * step, 2) for step in range(1, 11))

# Comparators in the main figure and in the win counts.
MAIN_COMPARATORS = ("Weighted kNN", "SVD", "ALRA", "SAVER", "MAGIC", "scVI", "scGPT")
# scGCL is reported only in the supplement: its default learning rate diverged,
# and the retained run uses a learning rate of 1e-6.
SUPPLEMENTARY_COMPARATORS = ("scGCL",)
# Inductive MAGIC and scVI are Safe Fusion teachers, fitted without the test
# cells. Their untrained rankings are reported apart from the main figure,
# whose MAGIC and scVI are the standard transductive baselines.
TEACHER_COMPARATORS = ("MAGIC (inductive)", "scVI (inductive)")
# Comparators whose values exist for the selector-fitting cells.
STACKED_COMPARATORS = (
    "Gene median", "SVD", "Weighted kNN", "ALRA", "MAGIC", "scVI", "SAVER", "scGPT",
    *TEACHER_COMPARATORS,
)

# Conversion from a contract's stored scale to the count scale of the masked
# input, given each cell's masked library size.
COUNT_SCALE = {
    "counts": lambda value, library: value,
    "normalized_expression_1e4": lambda value, library: value * (library / 1e4),
    "log1p_cpm": lambda value, library: np.expm1(value) * (library / 1e4),
}
# Frozen scGPT scores are within-cell binned expression values with no count
# scale, so they are ranked as scored.
NATIVE_SCALES = {"frozen_scgpt_masked_value_score"}


def fraction_name(fraction: float) -> str:
    return str(round(float(fraction), 2)).replace(".", "p")


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


@dataclass
class Unit:
    dataset: str
    key: str
    corrupted: Path
    coordinates: Path
    splits: Path
    truth: Path
    fit_split: str
    unit_column: str
    tie_seed: int
    selector_dir: Path
    contracts: dict[str, Path] = field(default_factory=dict)
    fit_cells: int | None = None


def add_root_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE)
    parser.add_argument("--colon-methods-root", type=Path, default=COLON_METHODS)
    parser.add_argument(
        "--baselines-root",
        type=Path,
        default=None,
        help="Defaults to <evidence-root>/baselines.",
    )


def build_units(evidence: Path, colon_methods: Path, baselines: Path | None = None, seed: int = 1729) -> list[Unit]:
    baselines = baselines or evidence / "baselines"
    units = []
    for fold in range(3):
        fold_root = evidence / "pancreas_crossfit" / f"fold_{fold}"
        units.append(
            Unit(
                dataset="Pancreas",
                key=f"pancreas_{fold}",
                corrupted=PANCREAS_RUN / "data" / "pancreas_islets" / "corrupted" / "mask_010.h5ad",
                coordinates=PANCREAS_RUN / "data" / "pancreas_islets" / "coordinates" / "mask_010.parquet",
                splits=fold_root / "splits.parquet",
                truth=PANCREAS_RUN / "data" / "pancreas_islets" / "preprocessed.h5ad",
                fit_split="development",
                unit_column="donor",
                tie_seed=seed + 100 * fold,
                selector_dir=fold_root / "selector_mlp_biology_range_fullteachers",
                contracts={
                    "Gene median": fold_root / "gene_median",
                    "Weighted kNN": fold_root / "graph_smooth",
                    "SVD": fold_root / "svd_impute",
                    "MAGIC (inductive)": fold_root / "magic_inductive",
                    "scVI (inductive)": fold_root / "scvi_inductive",
                    "ALRA": fold_root / "alra",
                    "SAVER": baselines / "saver" / "pancreas_mask_010",
                    "MAGIC": baselines / "magic" / "pancreas",
                    "scVI": baselines / "scvi" / "pancreas",
                    "scGCL": baselines / "scgcl" / "pancreas",
                    "scGPT": baselines / "scgpt_mvc" / "pancreas",
                },
            )
        )
    units.append(
        Unit(
            dataset="Colon",
            key="colon",
            corrupted=COLON_RUN / "data" / "colon_epithelial" / "corrupted" / "mask_010.h5ad",
            coordinates=COLON_RUN / "data" / "colon_epithelial" / "coordinates" / "mask_010.parquet",
            splits=COLON_RUN / "data" / "colon_epithelial" / "splits.parquet",
            truth=COLON_RUN / "data" / "colon_epithelial" / "preprocessed.h5ad",
            fit_split="validation",
            fit_cells=3852,
            unit_column="donor",
            tie_seed=seed + 1000,
            selector_dir=evidence / "selector_mlp_biology_range" / "colon",
            contracts={
                "Gene median": colon_methods / "gene_median",
                "Weighted kNN": colon_methods / "graph_smooth",
                "SVD": colon_methods / "svd_impute",
                "MAGIC (inductive)": colon_methods / "magic_inductive",
                "scVI (inductive)": colon_methods / "scvi_inductive",
                "ALRA": baselines / "alra" / "colon_mask_010",
                "SAVER": baselines / "saver" / "colon_mask_010",
                "MAGIC": baselines / "magic" / "colon",
                "scVI": baselines / "scvi" / "colon",
                "scGCL": baselines / "scgcl" / "colon",
                "scGPT": baselines / "scgpt_mvc" / "colon",
            },
        )
    )
    norman = evidence / "norman_crispra"
    units.append(
        Unit(
            dataset="CRISPRa",
            key="norman_crispra",
            corrupted=norman / "corrupted.h5ad",
            coordinates=norman / "coordinates.parquet",
            splits=norman / "splits.parquet",
            truth=ROOT / "external_data" / "prepared" / "norman_crispra.h5ad",
            fit_split="development",
            unit_column="target",
            tie_seed=seed + 2000,
            selector_dir=evidence / "selector_mlp_biology_range_fullteachers" / "norman_crispra",
            contracts={
                "Gene median": norman / "methods" / "gene_median",
                "Weighted kNN": norman / "methods" / "graph_smooth",
                "SVD": norman / "methods" / "svd_impute",
                "MAGIC (inductive)": norman / "methods" / "magic_inductive",
                "scVI (inductive)": norman / "methods" / "scvi_inductive",
                "ALRA": baselines / "alra" / "norman_mask_010",
                "SAVER": baselines / "saver" / "norman_full_mask_010",
                "MAGIC": baselines / "magic" / "norman",
                "scVI": baselines / "scvi" / "norman",
                "scGCL": baselines / "scgcl" / "norman",
                "scGPT": baselines / "scgpt_mvc" / "norman",
            },
        )
    )
    return units


def units_from_args(args: argparse.Namespace, seed: int = 1729) -> list[Unit]:
    return build_units(args.evidence_root, args.colon_methods_root, args.baselines_root, seed)


@dataclass
class UnitData:
    counts: np.ndarray
    split: np.ndarray
    unit_labels: np.ndarray  # biological unit of each cell
    library: np.ndarray  # masked library size of each cell
    masked: np.ndarray  # boolean masked-positive matrix
    cell_ids: list[str]
    gene_ids: list[str]


def load_unit(unit: Unit) -> UnitData:
    adata = ad.read_h5ad(unit.corrupted)
    counts = dense(adata.layers["corrupted_counts"]).astype(np.float32)
    split = (
        pd.read_parquet(unit.splits)
        .set_index("cell_id")
        .loc[adata.obs_names.astype(str), "split"]
        .to_numpy()
    )
    coordinates = pd.read_parquet(unit.coordinates)
    masked = np.zeros(counts.shape, dtype=bool)
    masked[
        coordinates["cell_index"].to_numpy(dtype=int),
        coordinates["gene_index"].to_numpy(dtype=int),
    ] = True
    return UnitData(
        counts=counts,
        split=split,
        unit_labels=adata.obs[unit.unit_column].astype(str).to_numpy(),
        library=counts.sum(axis=1, dtype=np.float64),
        masked=masked,
        cell_ids=adata.obs_names.astype(str).tolist(),
        gene_ids=adata.var_names.astype(str).tolist(),
    )


def contract_metadata(contract: Path, data: UnitData) -> dict:
    metadata = json.loads((contract / "metadata.json").read_text())
    if metadata["cell_ids"] != data.cell_ids:
        raise ValueError(f"Cell order differs for {contract}")
    if metadata["gene_ids"] != data.gene_ids:
        raise ValueError(f"Gene order differs for {contract}")
    return metadata


def stored_scale(metadata: dict) -> str:
    if "scale" in metadata:
        return metadata["scale"]
    # The scGCL adapter (run_scgcl_baseline.py) writes no scale field; its
    # postprocessing rescales the reconstruction to the corrupted library mass.
    if str(metadata.get("method", "")).startswith("scgcl") and "library mass" in metadata.get("postprocessing", ""):
        return "counts"
    raise ValueError(f"contract metadata for {metadata.get('method')} records no scale")


def count_scale_values(contract: Path, data: UnitData, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a contract's values at (rows, cols) on the count scale of the masked input."""
    metadata = contract_metadata(contract, data)
    scale = stored_scale(metadata)
    mean = np.load(contract / "mean.npy", mmap_mode="r", allow_pickle=False)
    values = np.asarray(mean[rows, cols], dtype=np.float64)
    if scale in NATIVE_SCALES:
        return values.astype(np.float32), scale
    if scale not in COUNT_SCALE:
        raise ValueError(f"No count-scale rule for scale {scale!r} in {contract}")
    return COUNT_SCALE[scale](values, data.library[rows]).astype(np.float32), scale


def unit_counts(
    selected: np.ndarray,
    labels: np.ndarray,
    unit_labels: np.ndarray,
) -> pd.DataFrame:
    """Selected entries, true positives and masked positives per biological unit."""
    frame = pd.DataFrame({"unit": unit_labels, "selected": selected, "labels": labels})
    frame["true_positive"] = frame["selected"] & frame["labels"].astype(bool)
    grouped = frame.groupby("unit", sort=True).agg(
        n_selected=("selected", "sum"),
        n_true_positive=("true_positive", "sum"),
        n_masked_positives=("labels", "sum"),
        n_zeros=("labels", "size"),
    )
    return grouped.reset_index()
