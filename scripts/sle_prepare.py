#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

TREG = "T4_reg"
LINEAGE = {
    "T4": "CD4_T", "T8": "CD8_T", "NK": "NK", "B": "B", "PB": "Plasmablast",
    "cM": "Monocyte", "ncM": "Monocyte", "cDC": "DC", "pDC": "DC", "Progen": "Other", "Prolif": "Other",
}
OBS_COLUMNS = (
    "Processing_Cohort", "author_cell_type", "ct_cov", "disease", "disease_state", "donor_id",
    "self_reported_ethnicity", "sex", "sample_uuid", "library_uuid", "development_stage",
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()

def decode(values) -> np.ndarray:
    return np.asarray([value.decode() if isinstance(value, bytes) else value for value in values], dtype=object)

def read_column(node) -> np.ndarray:
    if isinstance(node, h5py.Group):
        categories = decode(node["categories"][:])
        codes = node["codes"][:]
        return np.where(codes >= 0, categories[np.maximum(codes, 0)], None)
    return decode(node[:])

def read_rows(group: h5py.Group, rows: np.ndarray, n_genes: int) -> sparse.csr_matrix:
    indptr = group["indptr"][:]
    data_parts, index_parts, lengths = [], [], []
    for row in rows:
        start, stop = int(indptr[row]), int(indptr[row + 1])
        data_parts.append(group["data"][start:stop])
        index_parts.append(group["indices"][start:stop])
        lengths.append(stop - start)
    new_indptr = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    matrix = sparse.csr_matrix(
        (np.concatenate(data_parts), np.concatenate(index_parts), new_indptr), shape=(len(rows), n_genes)
    )
    matrix.sort_indices()
    return matrix

def choose_donors(donors: pd.DataFrame, column: str, per_stratum: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    chosen = []
    for _, block in donors.groupby(column, sort=True):
        names = np.asarray(sorted(block.index))
        if len(names) < per_stratum:
            raise ValueError(f"{column} stratum has {len(names)} donors, fewer than {per_stratum}")
        chosen.extend(sorted(names[rng.permutation(len(names))[:per_stratum]]))
    return donors.loc[chosen]

def assign_folds(donors: pd.DataFrame, column: str, folds: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, block in donors.groupby(column, sort=True):
        names = np.asarray(sorted(block.index))[rng.permutation(len(block))]
        for offset, donor in enumerate(names):
            assignment[str(donor)] = offset % folds
    return assignment

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="external_data/cellxgene/c55dc602-d168-4d15-acc1-5de4f2f5d551.h5ad")
    parser.add_argument("--config", default="configs/sle_treg_genes.json")
    parser.add_argument("--source-sha256", default=None, help="Fail unless the source file has this SHA-256.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-dir", required=True)
    parser.add_argument("--cohort", required=True, help="Processing_Cohort value, for example 4.0")
    parser.add_argument("--ancestry", default="European American")
    parser.add_argument("--disease", nargs="+", default=["systemic lupus erythematosus", "normal"])
    parser.add_argument("--contrast", choices=["condition", "sex"], required=True,
                        help="Donor attribute that the set contrasts; donors are chosen and folds stratified by it.")
    parser.add_argument("--donors-per-stratum", type=int, required=True)
    parser.add_argument("--cap-per-type", type=int, default=100,
                        help="Maximum cells per donor and author cell type other than Treg.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = Path(args.input)
    digest = sha256_file(source)
    if args.source_sha256 and digest != args.source_sha256:
        raise ValueError(f"source SHA-256 {digest} differs from {args.source_sha256}")

    with h5py.File(source, "r") as handle:
        obs = pd.DataFrame({name: read_column(handle["obs"][name]) for name in OBS_COLUMNS})
        obs.index = decode(handle["obs"][handle["obs"].attrs["_index"]][:])
        feature_name = read_column(handle["raw/var/feature_name"])
        gene_id = read_column(handle["raw/var/_index"])
        n_genes = len(gene_id)

        cohort = obs["Processing_Cohort"] == args.cohort
        eligible = cohort & (obs["self_reported_ethnicity"] == args.ancestry) & obs["disease"].isin(args.disease)
        cohort_obs = obs.loc[cohort]
        library_table = pd.crosstab(cohort_obs["library_uuid"], cohort_obs["disease"])
        frame = obs.loc[eligible].copy()
        frame["condition"] = np.where(frame["disease"] == "normal", "healthy", "SLE")
        per_donor = frame.groupby("donor_id").agg(
            condition=("condition", "first"), sex=("sex", "first"), n_conditions=("condition", "nunique"),
            n_sexes=("sex", "nunique"), samples=("sample_uuid", "nunique"),
        )
        if (per_donor[["n_conditions", "n_sexes", "samples"]] > 1).any().any():
            raise ValueError("a donor has more than one condition, sex or sample in the chosen cohort")
        donors = choose_donors(per_donor, args.contrast, args.donors_per_stratum, args.seed)
        frame = frame.loc[frame["donor_id"].isin(donors.index)].copy()
        frame["cell_type"] = frame["ct_cov"].where(frame["ct_cov"].notna(), frame["author_cell_type"]).astype(str)
        is_treg = frame["ct_cov"] == TREG
        frame["lineage"] = np.where(is_treg, "Treg", frame["author_cell_type"].map(LINEAGE))

        rng = np.random.default_rng(args.seed)
        keep = []
        for (donor, cell_type), block in frame.groupby(["donor_id", "cell_type"], sort=True):
            names = block.index.to_numpy()
            if cell_type != TREG and len(names) > args.cap_per_type:
                names = names[np.sort(rng.choice(len(names), size=args.cap_per_type, replace=False))]
            keep.extend(names.tolist())
        position = pd.Series(np.arange(len(obs)), index=obs.index)
        rows = np.sort(position.loc[keep].to_numpy())
        counts = read_rows(handle["raw/X"], rows, n_genes)

    if np.any(counts.data < 0) or np.any(counts.data != np.round(counts.data)):
        raise ValueError("source raw matrix is not integer counts")
    selected = obs.iloc[rows].copy()
    frame = frame.loc[selected.index]
    names = pd.Series(feature_name.astype(str))
    duplicated = names.duplicated(keep=False).to_numpy()
    var_names = np.where(duplicated, names + "_" + pd.Series(gene_id.astype(str)), names)
    var = pd.DataFrame({"gene_id": gene_id.astype(str), "feature_name": feature_name.astype(str)}, index=var_names)

    prepared_obs = pd.DataFrame(index=selected.index.astype(str))
    prepared_obs["donor"] = frame["donor_id"].astype(str).to_numpy()
    prepared_obs["condition"] = frame["condition"].to_numpy()
    prepared_obs["sex"] = frame["sex"].astype(str).to_numpy()
    prepared_obs["disease_state"] = frame["disease_state"].astype(str).to_numpy()
    prepared_obs["cell_type"] = frame["cell_type"].to_numpy()
    prepared_obs["author_cell_type"] = frame["author_cell_type"].astype(str).to_numpy()
    prepared_obs["lineage"] = frame["lineage"].astype(str).to_numpy()
    prepared_obs["is_treg"] = (frame["ct_cov"] == TREG).to_numpy()
    prepared_obs["library_uuid"] = frame["library_uuid"].astype(str).to_numpy()
    prepared_obs["processing_cohort"] = frame["Processing_Cohort"].astype(str).to_numpy()
    prepared_obs["ancestry"] = frame["self_reported_ethnicity"].astype(str).to_numpy()
    prepared_obs["development_stage"] = frame["development_stage"].astype(str).to_numpy()
    prepared_obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    counts = counts.astype(np.float32)
    prepared = ad.AnnData(X=counts, obs=prepared_obs, var=var)
    prepared.layers["counts"] = counts.copy()
    prepared.uns["source"] = {
        "file": str(source), "sha256": digest, "doi": "10.1126/science.abf1970",
        "cellxgene_collection_id": "436154da-bcf1-4130-9c8b-120ff9a888f2",
        "cellxgene_dataset_id": "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
        "cellxgene_dataset_version_id": source.stem,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output, compression="gzip")

    assignment = assign_folds(donors, args.contrast, args.folds, args.seed)
    fold_dir = Path(args.fold_dir)
    fold_details = []
    cell_donors = prepared_obs["donor"].to_numpy()
    for fold in range(args.folds):
        heldout = sorted(donor for donor, value in assignment.items() if value == fold)
        split = np.where(np.isin(cell_donors, heldout), "test", "development")
        path = fold_dir / f"fold_{fold}" / "splits.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "cell_id": prepared_obs.index, "biological_unit": cell_donors,
            "condition": prepared_obs["condition"].to_numpy(), "split": split, "fold": fold,
        }).to_parquet(path, index=False)
        fold_details.append({
            "fold": fold, "splits": str(path),
            "test_donors": donors.loc[heldout, [args.contrast]].reset_index().to_dict(orient="records"),
            "test_cells": int((split == "test").sum()), "development_cells": int((split == "development").sum()),
        })

    library_share = library_table.div(library_table.sum(axis=1), axis=0)
    chosen_libraries = pd.crosstab(prepared_obs["library_uuid"], prepared_obs[args.contrast])
    shared = (chosen_libraries > 0).all(axis=1) if chosen_libraries.shape[1] > 1 else chosen_libraries.iloc[:, 0] < 0
    config = json.loads(Path(args.config).read_text())
    sex_expression = pd.DataFrame(index=sorted(set(cell_donors)))
    library = pd.Series(np.asarray(counts.sum(axis=1)).ravel(), index=cell_donors).groupby(level=0).sum()
    for label, genes in (("xist", config["sex_x"]), ("y_genes", config["sex_y"])):
        columns = np.flatnonzero(np.isin(var.index, genes))
        total = pd.Series(np.asarray(counts[:, columns].sum(axis=1)).ravel(), index=cell_donors).groupby(level=0).sum()
        sex_expression[f"{label}_cp10k"] = (1e4 * total / library).reindex(sex_expression.index)
    sex_expression["metadata_sex"] = donors["sex"].reindex(sex_expression.index)
    sex_expression["expression_sex"] = np.where(
        sex_expression["y_genes_cp10k"] > sex_expression["xist_cp10k"], "male", "female")
    report = {
        "source": prepared.uns["source"],
        "design": {
            "processing_cohort": args.cohort, "ancestry": args.ancestry, "disease": args.disease,
            "contrast": args.contrast, "donors_per_stratum": args.donors_per_stratum,
            "cap_per_type": args.cap_per_type, "folds": args.folds, "seed": args.seed,
        },
        "eligible_donors": per_donor.groupby([args.contrast]).size().to_dict(),
        "eligible_donors_by_condition_and_sex": {
            f"{c}|{s}": int(n) for (c, s), n in per_donor.groupby(["condition", "sex"]).size().items()
        },
        "chosen_donors": donors[["condition", "sex"]].reset_index().to_dict(orient="records"),
        "cohort_libraries": int(len(library_table)),
        "cohort_library_disease_share_range": {
            column: [float(library_share[column].min()), float(library_share[column].max())]
            for column in library_share.columns
        },
        "cohort_libraries_with_both_conditions": int((library_table > 0).all(axis=1).sum()),
        "chosen_libraries": int(len(chosen_libraries)),
        "chosen_libraries_with_both_contrast_groups": int(shared.sum()),
        "chosen_cells_in_libraries_with_both_contrast_groups": float(
            prepared_obs["library_uuid"].isin(chosen_libraries.index[shared]).mean()),
        "sex_by_expression": sex_expression.reset_index(names="donor").to_dict(orient="records"),
        "sex_metadata_agrees_with_expression": bool(
            (sex_expression["metadata_sex"] == sex_expression["expression_sex"]).all()),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "n_cells": int(prepared.n_obs),
        "n_genes": int(prepared.n_vars),
        "cells_by_lineage": prepared_obs["lineage"].value_counts().to_dict(),
        "cells_by_lineage_and_condition": {
            f"{a}|{b}": int(n) for (a, b), n in prepared_obs.groupby(["lineage", "condition"]).size().items()
        },
        "treg_share": float(prepared_obs["is_treg"].mean()),
        "tregs_per_donor": prepared_obs.loc[prepared_obs["is_treg"]].groupby("donor").size().describe().to_dict(),
        "cells_per_donor": prepared_obs.groupby("donor").size().describe().to_dict(),
        "disease_state": prepared_obs["disease_state"].value_counts().to_dict(),
        "folds": fold_details,
    }
    report_path = fold_dir / "prepare_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))

if __name__ == "__main__":
    main()
