#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from sle_build_fold import variable_genes
from sle_prepare import decode, read_column, sha256_file

OBS_COLUMNS = (
    "Processing_Cohort", "author_cell_type", "ct_cov", "disease", "donor_id",
    "self_reported_ethnicity", "sex", "library_uuid",
)

def read_rows(group: h5py.Group, rows: np.ndarray, n_genes: int, window: int) -> sparse.csr_matrix:

    indptr = group["indptr"][:]
    data_parts, index_parts, lengths = [], [], []
    start = 0
    while start < len(rows):
        stop = start + int(np.searchsorted(rows[start:], rows[start] + window))
        block = rows[start:stop]
        first, last = int(indptr[block[0]]), int(indptr[block[-1] + 1])
        data = group["data"][first:last]
        indices = group["indices"][first:last]
        for row in block:
            a, b = int(indptr[row]) - first, int(indptr[row + 1]) - first
            data_parts.append(data[a:b])
            index_parts.append(indices[a:b].astype(np.int32))
            lengths.append(b - a)
        start = stop
    matrix = sparse.csr_matrix(
        (np.concatenate(data_parts), np.concatenate(index_parts), np.concatenate([[0], np.cumsum(lengths)])),
        shape=(len(rows), n_genes),
    )
    matrix.sort_indices()
    return matrix

def split_donors(donors: pd.DataFrame, test_fraction: float, seed: int) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    assignment = {}
    for _, block in donors.groupby("condition", sort=True):
        names = np.asarray(sorted(block.index))[rng.permutation(len(block))]
        n_test = int(round(test_fraction * len(names)))
        assignment.update({str(name): "test" for name in names[:n_test]})
        assignment.update({str(name): "development" for name in names[n_test:]})
    return assignment

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="external_data/cellxgene/c55dc602-d168-4d15-acc1-5de4f2f5d551.h5ad")
    parser.add_argument("--source-sha256", default="3c0b74d54c03838a49817edce95314e6a4fa048ef35b74d1e697b8b2fd07cc03")
    parser.add_argument("--cohort", default="4.0", help="Processing_Cohort value")
    parser.add_argument("--sizes", type=int, nargs="+", default=[25_000, 50_000, 100_000, 200_000])
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Share of donors of each disease status held out")
    parser.add_argument("--variable-genes", type=int, default=2000)
    parser.add_argument("--read-window", type=int, default=20_000, help="Stored rows read from the source in one slice")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_evidence/review_round3/scale"))
    args = parser.parse_args()

    source = Path(args.input)
    digest = sha256_file(source)
    if digest != args.source_sha256:
        raise ValueError(f"source SHA-256 {digest} differs from {args.source_sha256}")
    sizes = sorted(args.sizes)

    with h5py.File(source, "r") as handle:
        obs = pd.DataFrame({name: read_column(handle["obs"][name]) for name in OBS_COLUMNS})
        obs.index = decode(handle["obs"][handle["obs"].attrs["_index"]][:])
        feature_name = read_column(handle["raw/var/feature_name"]).astype(str)
        gene_id = read_column(handle["raw/var/_index"]).astype(str)
        cohort_rows = np.flatnonzero((obs["Processing_Cohort"] == args.cohort).to_numpy())
        if sizes[-1] > len(cohort_rows):
            raise ValueError(f"cohort {args.cohort} has {len(cohort_rows)} cells, fewer than {sizes[-1]}")
        cohort = obs.iloc[cohort_rows].copy()
        cohort["condition"] = np.where(cohort["disease"] == "normal", "healthy", "SLE")
        per_donor = cohort.groupby("donor_id").agg(
            condition=("condition", "first"), n_conditions=("condition", "nunique"), cells=("condition", "size"),
            sex=("sex", "first"), ancestry=("self_reported_ethnicity", "first"),
        )
        if (per_donor["n_conditions"] > 1).any():
            raise ValueError("a donor has more than one condition in the cohort")
        assignment = split_donors(per_donor, args.test_fraction, args.seed)
        largest = cohort_rows[np.random.default_rng(args.seed).permutation(len(cohort_rows))[: sizes[-1]]]
        stored = np.sort(largest)
        counts = read_rows(handle["raw/X"], stored, len(gene_id), args.read_window)

    if np.any(counts.data < 0) or np.any(counts.data != np.round(counts.data)):
        raise ValueError("source raw matrix is not integer counts")
    counts = counts[np.searchsorted(stored, largest)]
    cells = obs.iloc[largest].copy()
    cells["condition"] = np.where(cells["disease"] == "normal", "healthy", "SLE")
    cells["split"] = cells["donor_id"].astype(str).map(assignment).to_numpy()

    development = (cells["split"] == "development").to_numpy()
    selected, score = variable_genes(counts, development, args.variable_genes)
    selected = np.sort(selected)
    names = pd.Series(feature_name)
    duplicated = names.duplicated(keep=False).to_numpy()
    var_names = np.where(duplicated, names + "_" + pd.Series(gene_id), names)
    var = pd.DataFrame({"gene_id": gene_id[selected], "feature_name": feature_name[selected],
                        "selection_score": score[selected]}, index=var_names[selected])
    gene_counts = counts[:, selected].tocsr().astype(np.float32)

    prepared_obs = pd.DataFrame(index=cells.index.astype(str))
    prepared_obs["donor"] = cells["donor_id"].astype(str).to_numpy()
    prepared_obs["condition"] = cells["condition"].to_numpy()
    prepared_obs["cell_type"] = cells["ct_cov"].where(cells["ct_cov"].notna(), cells["author_cell_type"]).astype(str).to_numpy()
    prepared_obs["library_uuid"] = cells["library_uuid"].astype(str).to_numpy()
    prepared_obs["total_counts_all_genes"] = np.asarray(counts.sum(axis=1)).ravel()
    split = cells["split"].to_numpy()

    report = {
        "source": {"file": str(source), "sha256": digest, "doi": "10.1126/science.abf1970",
                   "cellxgene_dataset_version_id": source.stem, "matrix": "raw/X (UMI counts)"},
        "design": {"processing_cohort": args.cohort, "sizes": sizes, "test_fraction": args.test_fraction,
                   "variable_genes": args.variable_genes, "seed": args.seed,
                   "gene_rule": "(variance - mean) / mean of raw counts among genes detected in at least max(5, 0.5%) "
                                "of the training-donor cells of the largest subset"},
        "cohort_cells": int(len(cohort_rows)),
        "cohort_donors": int(len(per_donor)),
        "donors": per_donor.assign(split=[assignment[str(name)] for name in per_donor.index])
                           .reset_index()[["donor_id", "condition", "sex", "ancestry", "cells", "split"]]
                           .to_dict(orient="records"),
        "donors_by_split_and_condition": {
            f"{s}|{c}": int(n) for (s, c), n in
            pd.crosstab(pd.Series(assignment), per_donor["condition"].reindex(list(assignment))).stack().items()
        },
        "genes": var.index.tolist(),
        "subsets": [],
    }
    for size in sizes:
        rows = slice(0, size)
        directory = args.output_dir / f"cells_{size}"
        directory.mkdir(parents=True, exist_ok=True)
        subset = gene_counts[rows]
        adata = ad.AnnData(X=subset.copy(), obs=prepared_obs.iloc[rows].copy(), var=var.copy())
        adata.layers["counts"] = subset.copy()
        adata.obs["total_counts"] = np.asarray(subset.sum(axis=1)).ravel()
        adata.uns["source"] = report["source"]
        adata.write_h5ad(directory / "truth.h5ad")
        pd.DataFrame({
            "cell_id": adata.obs_names.astype(str), "biological_unit": adata.obs["donor"].to_numpy(),
            "condition": adata.obs["condition"].to_numpy(), "split": split[rows],
        }).to_parquet(directory / "splits.parquet", index=False)
        subset_split = split[rows]
        report["subsets"].append({
            "cells": int(size), "genes": int(subset.shape[1]),
            "development_cells": int((subset_split == "development").sum()),
            "test_cells": int((subset_split == "test").sum()),
            "development_donors": int(adata.obs.loc[subset_split == "development", "donor"].nunique()),
            "test_donors": int(adata.obs.loc[subset_split == "test", "donor"].nunique()),
            "nonzero_fraction": float(subset.nnz / np.prod(subset.shape)),
            "median_total_counts": float(np.median(adata.obs["total_counts"])),
            "test_cells_per_donor": adata.obs.loc[subset_split == "test"].groupby("donor", observed=True).size().describe().to_dict(),
            "truth": str(directory / "truth.h5ad"), "truth_sha256": sha256_file(directory / "truth.h5ad"),
            "splits": str(directory / "splits.parquet"),
        })
    (args.output_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({key: report[key] for key in ("cohort_cells", "cohort_donors", "donors_by_split_and_condition", "subsets")},
                     indent=2, default=str))

if __name__ == "__main__":
    main()
