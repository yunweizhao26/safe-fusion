#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse

LINEAGE = {"CD4 T": "CD4_T", "CD8 T": "CD8_T", "NK": "NK", "B": "B", "Mono": "Monocyte", "DC": "DC",
           "other T": "Other", "other": "Other"}

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()

def read_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return [line.rstrip("\n").split("\t")[0] for line in handle]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("external_data/citeseq_hao2021"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adt-output", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--cap-per-type", type=int, default=100)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = args.source_dir
    meta = pd.read_csv(source / "GSE164378_sc.meta.data_3P.csv.gz", index_col=0)
    barcodes = read_lines(source / "GSM5008737_RNA_3P-barcodes.tsv.gz")
    if barcodes != read_lines(source / "GSM5008738_ADT_3P-barcodes.tsv.gz") or barcodes != meta.index.tolist():
        raise ValueError("RNA, ADT and metadata barcodes differ")
    keep_meta = meta.loc[meta["celltype.l2"] != "Doublet"].copy()
    keep_meta["cell_type"] = keep_meta["celltype.l2"].astype(str)
    keep_meta["donor"] = keep_meta["donor"].astype(str)
    rng = np.random.default_rng(args.seed)
    keep = []
    for (_, cell_type), block in keep_meta.groupby(["donor", "cell_type"], sort=True):
        names = block.index.to_numpy()
        if cell_type != "Treg" and len(names) > args.cap_per_type:
            names = names[np.sort(rng.choice(len(names), size=args.cap_per_type, replace=False))]
        keep.extend(names.tolist())
    position = pd.Series(np.arange(len(barcodes)), index=barcodes)
    columns = np.sort(position.loc[keep].to_numpy())
    cells = [barcodes[index] for index in columns]

    rna = sparse.csc_matrix(io.mmread(source / "GSM5008737_RNA_3P-matrix.mtx.gz"))
    counts = rna[:, columns].T.tocsr().astype(np.float32)
    del rna
    adt = sparse.csc_matrix(io.mmread(source / "GSM5008738_ADT_3P-matrix.mtx.gz"))[:, columns].T.toarray().astype(np.float64)
    genes = read_lines(source / "GSM5008737_RNA_3P-features.tsv.gz")
    tags = read_lines(source / "GSM5008738_ADT_3P-features.tsv.gz")
    names = pd.Series(genes)
    var_names = np.where(names.duplicated(keep=False), names + "_" + names.groupby(names).cumcount().astype(str), names)

    obs = keep_meta.loc[cells]
    prepared_obs = pd.DataFrame(index=pd.Index(cells, name=None))
    prepared_obs["donor"] = obs["donor"].to_numpy()
    prepared_obs["condition"] = "healthy"
    prepared_obs["time"] = obs["time"].astype(str).to_numpy()
    prepared_obs["cell_type"] = obs["cell_type"].to_numpy()
    prepared_obs["lineage"] = np.where(obs["cell_type"] == "Treg", "Treg", obs["celltype.l1"].map(LINEAGE)).astype(str)
    prepared_obs["is_treg"] = (obs["cell_type"] == "Treg").to_numpy()
    prepared_obs["lane"] = obs["lane"].astype(str).to_numpy()
    prepared_obs["total_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    prepared = ad.AnnData(X=counts, obs=prepared_obs, var=pd.DataFrame(index=var_names))
    prepared.layers["counts"] = counts.copy()
    prepared.uns["source"] = {"geo": "GSE164378", "rna": "GSM5008737", "adt": "GSM5008738", "doi": "10.1016/j.cell.2021.04.048"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(args.output, compression="gzip")

    log_adt = np.log1p(adt)
    clr = log_adt - log_adt.mean(axis=1, keepdims=True)
    adt_frame = pd.DataFrame(adt, index=cells, columns=[f"count|{tag}" for tag in tags])
    adt_frame = pd.concat([adt_frame, pd.DataFrame(clr, index=cells, columns=[f"clr|{tag}" for tag in tags])], axis=1)
    adt_frame.index.name = "cell_id"
    args.adt_output.parent.mkdir(parents=True, exist_ok=True)
    adt_frame.reset_index().to_parquet(args.adt_output, index=False)

    donors = np.asarray(sorted(prepared_obs["donor"].unique()))
    donors = donors[np.random.default_rng(args.seed).permutation(len(donors))]
    assignment = {donor: index % args.folds for index, donor in enumerate(donors)}
    folds = []
    for fold in range(args.folds):
        heldout = sorted(donor for donor, value in assignment.items() if value == fold)
        split = np.where(prepared_obs["donor"].isin(heldout), "test", "development")
        path = args.fold_dir / f"fold_{fold}" / "splits.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cell_id": cells, "biological_unit": prepared_obs["donor"].to_numpy(),
                      "condition": prepared_obs["condition"].to_numpy(), "split": split, "fold": fold}).to_parquet(path, index=False)
        folds.append({"fold": fold, "test_donors": heldout, "test_cells": int((split == "test").sum())})
    fcrl3 = tags.index("CD307c/FcRL3")
    report = {
        "source_sha256": {path.name: sha256_file(path) for path in sorted(source.glob("*.gz"))},
        "output": str(args.output), "output_sha256": sha256_file(args.output),
        "cap_per_type": args.cap_per_type, "seed": args.seed, "n_cells": int(prepared.n_obs), "n_genes": int(prepared.n_vars),
        "cells_by_lineage": prepared_obs["lineage"].value_counts().to_dict(),
        "treg_share": float(prepared_obs["is_treg"].mean()),
        "tregs_per_donor": prepared_obs.loc[prepared_obs["is_treg"]].groupby("donor").size().to_dict(),
        "adt_tags": len(tags), "fcrl3_tag": tags[fcrl3],
        "fcrl3_adt_median_count_by_lineage": pd.Series(adt[:, fcrl3]).groupby(prepared_obs["lineage"].to_numpy()).median().to_dict(),
        "fcrl3_rna_detection_by_lineage": pd.Series(np.asarray(counts[:, genes.index("FCRL3")].todense()).ravel() > 0)
        .groupby(prepared_obs["lineage"].to_numpy()).mean().to_dict(),
        "folds": folds,
    }
    (args.fold_dir / "prepare_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
