#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from prepare_external_perturbseq import dataset_annotations

PREPARED = REPOSITORY / "external_data" / "prepared"
SCPERTURB = REPOSITORY / "external_data" / "scperturb"
CELLXGENE = REPOSITORY / "external_data" / "cellxgene"
SCREENS = {
    "adamson_crispri": ("adamson_crispri", SCPERTURB / "AdamsonWeissman2016_GSM2406681_10X010.h5ad"),
    "dixit_ko": ("dixit_ko", SCPERTURB / "DixitRegev2016.h5ad"),
    "papalexi_eccite": ("papalexi_eccite", SCPERTURB / "PapalexiSatija2021_eccite_RNA.h5ad"),
    "papalexi_eccite_crossmodal": ("papalexi_eccite", SCPERTURB / "PapalexiSatija2021_eccite_RNA.h5ad"),
}
TISSUES = {
    "colon_epithelial": (
        CELLXGENE / "63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad",
        {"donor": "donor_id", "cell_type": "Celltype", "cell_type_broad": "cell_type"},
        ("condition", "Type", lambda value: np.where(value == "Infl", "inflamed", "not_inflamed")),
    ),
    "pancreas_islets": (
        CELLXGENE / "f89a618b-fe4b-404e-bd39-7c574529b1f5.h5ad",
        {"donor": "donor_id", "cell_type": "cell_label", "cell_type_broad": "cell_type", "condition": "disease_state"},
        None,
    ),
}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def counts_of(adata: ad.AnnData) -> sparse.csr_matrix:
    return sparse.csr_matrix(adata.layers["counts"] if "counts" in adata.layers else adata.X)


def rows_equal(left: sparse.csr_matrix, right: sparse.csr_matrix) -> float:
    difference = abs(sparse.csr_matrix(left, dtype=np.float64) - sparse.csr_matrix(right, dtype=np.float64))
    return float(np.mean(np.asarray(difference.sum(axis=1)).ravel() == 0))


def target_change(counts: sparse.csr_matrix, genes: pd.Index, target: np.ndarray, control: np.ndarray) -> dict:
    changes = {}
    for name in sorted(set(target[~control]) - {"none"}):
        if name not in genes:
            continue
        column = dense(counts[:, genes.get_loc(name)]).ravel()
        changes[name] = float(np.log2(column[target == name].mean() + 1) - np.log2(column[control].mean() + 1))
    values = np.asarray(list(changes.values()))
    return {
        "targets_in_panel": int(len(values)),
        "median_log2fc": float(np.median(values)) if len(values) else None,
        "targets_below_zero": int(np.sum(values < 0)),
        "targets_at_or_below_minus_0.25": int(np.sum(values <= -0.25)),
        "targets_at_or_above_0.5": int(np.sum(values >= 0.5)),
    }


def audit_screen(name: str, dataset: str, source_path: Path) -> dict:
    prepared = ad.read_h5ad(PREPARED / f"{name}.h5ad")
    source = ad.read_h5ad(source_path)
    rows = source.obs_names.get_indexer(prepared.obs["source_cell_id"].astype(str))
    cols = source.var_names.get_indexer(prepared.var_names)
    targets, control = dataset_annotations(source, dataset)
    counts = counts_of(prepared)
    target = prepared.obs["target"].astype(str).to_numpy()
    is_control = prepared.obs["control"].astype(int).to_numpy() == 1
    return {
        "cells": int(prepared.n_obs),
        "source_rows_found": int(np.sum(rows >= 0)),
        "genes_found": int(np.sum(cols >= 0)),
        "cells_with_source_counts": rows_equal(counts, sparse.csr_matrix(source.X)[rows][:, cols]),
        "cells_with_source_target": float(np.mean(target == targets[rows])),
        "cells_with_source_control_flag": float(np.mean(is_control == control[rows])),
        "target_gene_change_in_labelled_cells": target_change(counts, prepared.var_names, target, is_control),
    }


def audit_papalexi_protein(panel_path: Path, mudata_path: Path) -> dict:
    import mudata as md

    prepared = ad.read_h5ad(PREPARED / "papalexi_eccite_crossmodal.h5ad")
    panel = pd.read_parquet(panel_path).drop_duplicates("cell_id").set_index("cell_id")
    source_ids = prepared.obs["source_cell_id"].astype(str)
    joined = source_ids.isin(panel.index).to_numpy()
    matched = panel.loc[source_ids[joined]]
    target = prepared.obs["target"].astype(str).to_numpy()[joined]
    panel_target = matched["target"].astype(str).to_numpy()
    perturbed = target != "none"
    rna = md.read_h5mu(mudata_path)["rna"]
    rna_rows = rna.obs_names.get_indexer(matched["mudata_cell_id"].astype(str))
    rna_cols = rna.var_names.get_indexer(prepared.var_names)
    shared = rna_cols >= 0
    prepared_counts = counts_of(prepared)[np.flatnonzero(joined)][:, np.flatnonzero(shared)]
    mudata_counts = sparse.csr_matrix(rna.X)[rna_rows][:, rna_cols[shared]]
    return {
        "prepared_cells": int(prepared.n_obs),
        "cells_joined_to_protein": int(joined.sum()),
        "mudata_rows_found": int(np.sum(rna_rows >= 0)),
        "genes_shared_with_mudata": int(shared.sum()),
        "perturbed_cells_with_panel_target": float(np.mean(target[perturbed] == panel_target[perturbed])),
        "panel_targets_of_control_cells": sorted(set(panel_target[~perturbed])),
        "cells_with_mudata_rna_counts": rows_equal(prepared_counts, mudata_counts),
    }


def audit_zebrafish(source_path: Path) -> dict:
    prepared = ad.read_h5ad(PREPARED / "zebrafish_trajectory.h5ad")
    source = ad.read_h5ad(source_path)
    rows = source.obs_names.get_indexer(prepared.obs["source_cell_id"].astype(str))
    cols = source.var_names.get_indexer(prepared.var_names)
    return {
        "cells": int(prepared.n_obs),
        "source_rows_found": int(np.sum(rows >= 0)),
        "cells_with_source_counts": rows_equal(counts_of(prepared), sparse.csr_matrix(source.X)[rows][:, cols]),
        "cells_with_source_stage": float(np.mean(
            prepared.obs["condition"].astype(str).to_numpy() == source.obs["Stage"].astype(str).to_numpy()[rows])),
    }


def audit_tissue(name: str, source_path: Path, columns: dict, derived) -> dict:
    prepared = ad.read_h5ad(PREPARED / f"{name}.h5ad")
    source = ad.read_h5ad(source_path, backed="r")
    rows = source.obs_names.get_indexer(prepared.obs_names)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("prepared cells are not in increasing source order")
    cols = source.raw.var_names.get_indexer(prepared.var_names)
    source_obs = source.obs.iloc[rows]
    result = {
        "cells": int(prepared.n_obs),
        "source_rows_found": int(np.sum(rows >= 0)),
        "cells_with_source_counts": rows_equal(counts_of(prepared), sparse.csr_matrix(source.raw.X[rows])[:, cols]),
    }
    for column, source_column in columns.items():
        result[f"cells_with_source_{column}"] = float(np.mean(
            prepared.obs[column].astype(str).to_numpy() == source_obs[source_column].astype(str).to_numpy()))
    if derived is not None:
        column, source_column, rule = derived
        result[f"cells_with_source_{column}"] = float(np.mean(
            prepared.obs[column].astype(str).to_numpy() == rule(source_obs[source_column].astype(str).to_numpy())))
    return result


def norman_source_rows(prepared: ad.AnnData, conditions: np.ndarray, control_flag: np.ndarray, report: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if "qc_all_conditions" in report:
        drawn = [row for row in report["qc_all_conditions"] if row["reason"] != "too_few_cells"]
    else:
        drawn = report["conditions"]
    cells: list[int] = []
    for row in sorted(drawn, key=lambda value: value["target"]):
        positions = np.flatnonzero(conditions == row["condition"])
        if len(positions) > report["qc_rule"]["max_cells_per_condition"]:
            positions = rng.choice(positions, size=report["qc_rule"]["max_cells_per_condition"], replace=False)
        cells.extend(positions.tolist())
    controls = np.flatnonzero(control_flag == 1)
    controls = rng.choice(controls, size=report["qc_rule"]["max_control_cells"], replace=False)
    cells = np.asarray(sorted(cells + controls.tolist()))
    labels = np.where(control_flag[cells] == 1, "ctrl", conditions[cells])
    kept = np.isin(labels, [*[row["condition"] for row in report["conditions"]], "ctrl"])
    return cells[kept]


def audit_norman(prepared_path: Path, source_path: Path) -> dict:
    prepared = ad.read_h5ad(prepared_path)
    report = json.loads(prepared_path.with_suffix(".report.json").read_text())
    with h5py.File(source_path, "r") as handle:
        conditions = handle["obs/__categories/condition"][:].astype(str)[handle["obs/condition"][:]]
        control_flag = handle["obs/control"][:]
        rows = norman_source_rows(prepared, conditions, control_flag, report, report["seed"])
        source_counts = sparse.csr_matrix(handle["layers/counts"][rows].astype(np.float32))
    source_labels = np.where(control_flag[rows] == 1, "ctrl", conditions[rows])
    genes = pd.Index(prepared.var["feature_name"].astype(str))
    condition = prepared.obs["condition"].astype(str).to_numpy()
    return {
        "cells": int(prepared.n_obs),
        "cells_with_source_counts": rows_equal(counts_of(prepared), source_counts),
        "cells_with_source_condition": float(np.mean(condition == source_labels)),
        "target_gene_change_in_labelled_cells": target_change(
            counts_of(prepared), genes, prepared.obs["target"].astype(str).to_numpy(), condition == "ctrl"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=REPOSITORY / "artifacts" / "paper_evidence" / "review_round2" / "label_audit" / "label_audit.json")
    parser.add_argument("--norman-source", type=Path, required=True, help="GEARS perturb_processed.h5ad")
    parser.add_argument("--norman-rebuilt", type=Path,
                        default=REPOSITORY / "artifacts" / "paper_evidence" / "review_round2" / "leakage_free" / "norman_crispra" / "prepared.h5ad")
    parser.add_argument("--zebrafish-source", type=Path,
                        default=REPOSITORY / "external_data" / "trajectory" / "zebrafish_embryogenesis_axial_mesoderm.h5ad")
    parser.add_argument("--papalexi-panel", type=Path,
                        default=REPOSITORY / "artifacts" / "paper_evidence" / "papalexi_crossmodal" / "audit" / "matched_rna_adt_panel.parquet")
    parser.add_argument("--papalexi-mudata", type=Path,
                        default=REPOSITORY / "external_data" / "papalexi_multimodal" / "papalexi.h5mu")
    parser.add_argument("--datasets", nargs="+", default=[
        "norman_rebuilt", *SCREENS, "papalexi_protein_join", "zebrafish_trajectory", *TISSUES])
    args = parser.parse_args()

    audit = json.loads(args.output.read_text()) if args.output.exists() else {}
    for name in args.datasets:
        if name == "norman_rebuilt":
            result = audit_norman(args.norman_rebuilt, args.norman_source)
        elif name in SCREENS:
            result = audit_screen(name, *SCREENS[name])
        elif name == "papalexi_protein_join":
            result = audit_papalexi_protein(args.papalexi_panel, args.papalexi_mudata)
        elif name == "zebrafish_trajectory":
            result = audit_zebrafish(args.zebrafish_source)
        else:
            result = audit_tissue(name, *TISSUES[name])
        audit[name] = result
        print(json.dumps({name: result}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2) + "\n")


if __name__ == "__main__":
    main()
