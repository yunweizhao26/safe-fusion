#!/usr/bin/env python3
"""Prepare donor-resolved GSE100501 RNA and antibody measurements.

Feature selection and protein proxy thresholds use donor 1 only. Donor 2 is
validation and donor 3 remains locked test. Antibody reagent IDs are retained
verbatim, including multiple reagents targeting the same gene.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


RNA_RE = re.compile(r"GSM\d+_mRNA_(\d+)_Donor_(\d+)_(no|with)_aCD27_matrix\.txt\.gz$")

PROTEIN_GENE = {
    "CD3": "CD3D", "CD4": "CD4", "CD5": "CD5", "CD7": "CD7",
    "CD8": "CD8A", "CD8a": "CD8A", "CD9": "CD9", "CD10": "MME",
    "CD11b": "ITGAM", "CD11c": "ITGAX", "CD13": "ANPEP", "CD14": "CD14",
    "CD15": "FUT4", "CD16": "FCGR3A", "CD19": "CD19", "CD20": "MS4A1",
    "CD22": "CD22", "CD23": "FCER2", "CD25": "IL2RA", "CD27": "CD27",
    "CD28": "CD28", "CD33": "CD33", "CD34": "CD34", "CD38": "CD38",
    "CD40": "CD40", "CD44": "CD44", "CD45": "PTPRC", "CD45RA": "PTPRC",
    "CD45RO": "PTPRC", "CD49b": "ITGA2", "CD56": "NCAM1", "CD57": "B3GAT1",
    "CD64": "FCGR1A", "CD66b": "CEACAM8", "CD68": "CD68", "CD69": "CD69",
    "CD70": "CD70", "CD73": "NT5E", "CD117": "KIT", "CD123": "IL3RA",
    "CD127": "IL7R", "CD137": "TNFRSF9", "CD138": "SDC1", "CD141": "THBD",
    "CD152": "CTLA4", "CD154": "CD40LG", "CD155": "PVR", "CD160": "CD160",
    "CD163": "CD163", "CD197": "CCR7", "CD223": "LAG3", "CD272": "BTLA",
    "CD273": "PDCD1LG2", "CD274": "CD274", "CD278": "ICOS", "CD279": "PDCD1",
    "CD335": "NCR1", "CD357": "TNFRSF18", "CD366": "HAVCR2", "CD370": "CLEC9A",
    "EpCAM": "EPCAM", "FOXP3": "FOXP3", "HLA-DRA": "HLA-DRA",
    "HLA-ABC": "HLA-ABC", "HLA-E": "HLA-E", "HLA-G": "HLA-G",
    "TIGIT": "TIGIT",
}

RNA_MARKERS = {
    "CD3D", "CD3E", "TRAC", "CD4", "CD8A", "CD8B", "IL7R", "CCR7", "LTB",
    "CD19", "MS4A1", "CD79A", "CD37", "CD22", "CD14", "LYZ", "S100A8", "S100A9",
    "FCGR3A", "LST1", "CTSS", "TYROBP", "NCAM1", "NKG7", "GNLY", "KLRD1",
    "HLA-DRA", "CD274", "PDCD1LG2", "IL2RA", "CTLA4", "MKI67", "IFITM1", "IFITM3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_header(path: Path) -> list[str]:
    with gzip.open(path, "rt") as handle:
        return handle.readline().rstrip("\n").split("\t")


def sample_files(root: Path) -> list[dict[str, object]]:
    records = []
    for rna in sorted(root.glob("GSM*_mRNA_*_Donor_*_matrix.txt.gz")):
        match = RNA_RE.search(rna.name)
        if match is None:
            continue
        sample, donor, treatment = match.groups()
        protein_candidates = sorted(root.glob(f"GSM*_protein_{sample}_Donor_{donor}_{treatment}_aCD27_matrix.txt.gz"))
        if len(protein_candidates) != 1:
            raise ValueError(f"expected one matching protein file for {rna.name}, found {protein_candidates}")
        protein = protein_candidates[0]
        if matrix_header(rna) != matrix_header(protein):
            raise ValueError(f"RNA/protein cell order mismatch for sample {sample}")
        records.append({"sample": sample, "donor": donor, "treatment": treatment, "rna": rna, "protein": protein})
    if len(records) != 10:
        raise ValueError(f"expected 10 donor samples, found {len(records)}")
    return records


def choose_cells(records: list[dict[str, object]], maximum: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    for record in records:
        cells = np.asarray(matrix_header(record["rna"]), dtype=str)
        if len(cells) > maximum:
            positions = np.sort(rng.choice(len(cells), maximum, replace=False))
            cells = cells[positions]
        record["cells"] = cells.tolist()


def read_selected(path: Path, cells: list[str]) -> pd.DataFrame:
    # The GEO matrices have an intentionally blank index header. Pandas infers
    # the first field of each data row as the index.
    frame = pd.read_csv(path, sep="\t", index_col=0, usecols=lambda value: value in set(cells))
    return frame.loc[:, cells]


def feature_selection(records: list[dict[str, object]], n_features: int) -> list[str]:
    dev = [record for record in records if record["donor"] == "1"]
    totals: pd.Series | None = None
    totals_sq: pd.Series | None = None
    detected: pd.Series | None = None
    n_cells = 0
    for record in dev:
        frame = read_selected(record["rna"], record["cells"]).astype(np.float64)
        current_sum = frame.sum(axis=1)
        current_sq = frame.pow(2).sum(axis=1)
        current_detected = frame.gt(0).sum(axis=1)
        totals = current_sum if totals is None else totals.add(current_sum, fill_value=0)
        totals_sq = current_sq if totals_sq is None else totals_sq.add(current_sq, fill_value=0)
        detected = current_detected if detected is None else detected.add(current_detected, fill_value=0)
        n_cells += frame.shape[1]
    assert totals is not None and totals_sq is not None and detected is not None
    mean = totals / n_cells
    variance = (totals_sq / n_cells - mean.pow(2)).clip(lower=0)
    score = (variance - mean) / (mean + 1e-8)
    score[detected < max(5, int(0.005 * n_cells))] = -np.inf
    selected = score.sort_values(ascending=False, kind="stable").head(n_features).index.astype(str).tolist()
    available = set(score.index.astype(str))
    selected.extend(sorted((RNA_MARKERS | set(PROTEIN_GENE.values())) & available - set(selected)))
    return selected


def reagent_gene(reagent_id: str) -> str:
    marker = reagent_id.rsplit("_", 1)[0]
    return PROTEIN_GENE.get(marker, marker)


def canonical_indices(reagents: list[str], names: set[str]) -> list[int]:
    return [i for i, reagent in enumerate(reagents) if reagent.rsplit("_", 1)[0] in names]


def protein_proxy_labels(protein: np.ndarray, reagents: list[str], development: np.ndarray) -> tuple[np.ndarray, dict]:
    logp = np.log1p(protein.astype(np.float64))
    groups = {
        "T": {"CD3", "CD5", "CD7"},
        "B": {"CD19", "CD20", "CD22"},
        "NK": {"CD56", "CD335", "CD49b"},
        "monocyte": {"CD14", "CD11b", "CD33", "CD64"},
    }
    raw_scores = np.column_stack([
        logp[:, canonical_indices(reagents, markers)].mean(axis=1) for markers in groups.values()
    ])
    center = np.median(raw_scores[development], axis=0)
    scale = np.median(np.abs(raw_scores[development] - center), axis=0) * 1.4826
    scale[scale < 1e-6] = np.std(raw_scores[development], axis=0)[scale < 1e-6] + 1e-6
    standardized = (raw_scores - center) / scale
    winner = np.argmax(standardized, axis=1)
    labels = np.asarray(list(groups), dtype=object)[winner]
    threshold = float(np.quantile(np.max(standardized[development], axis=1), 0.10))
    labels[np.max(standardized, axis=1) < threshold] = "other"
    metadata = {
        "groups": {key: sorted(value) for key, value in groups.items()},
        "center": center.tolist(), "scale": scale.tolist(), "minimum_score": threshold,
        "fitted_on": "donor 1 development cells only", "interpretation": "protein-derived proxy, not certified identity",
    }
    return labels.astype(str), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protein-long", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--cells-per-sample", type=int, default=250)
    parser.add_argument("--variable-genes", type=int, default=1000)
    args = parser.parse_args()

    root = Path(args.input_dir).resolve()
    records = sample_files(root)
    choose_cells(records, args.cells_per_sample, args.seed)
    selected_genes = feature_selection(records, args.variable_genes)

    first_protein = read_selected(records[0]["protein"], records[0]["cells"])
    reagents = first_protein.index.astype(str).tolist()
    if len(reagents) != len(set(reagents)):
        raise ValueError("duplicate reagent IDs within the antibody panel")

    rna_blocks, protein_blocks, obs_blocks, long_blocks = [], [], [], []
    for record in records:
        cells = record["cells"]
        rna = read_selected(record["rna"], cells).reindex(selected_genes).fillna(0)
        protein = read_selected(record["protein"], cells)
        observed_reagents = protein.index.astype(str).tolist()
        if set(observed_reagents) != set(reagents):
            raise ValueError(f"antibody reagent set mismatch in {record['protein']}")
        # GEO sample 10 swaps two row positions; align by full reagent ID while
        # retaining every reagent as its own feature.
        protein = protein.reindex(reagents)
        unique_ids = [f"sample{record['sample']}:{cell}" for cell in cells]
        rna_blocks.append(sparse.csr_matrix(rna.to_numpy(dtype=np.int32).T))
        protein_blocks.append(protein.to_numpy(dtype=np.float32).T)
        split = {"1": "development", "2": "validation", "3": "test"}[record["donor"]]
        obs_blocks.append(pd.DataFrame({
            "donor": f"donor_{record['donor']}", "sample": f"sample_{record['sample']}",
            "condition": "aCD27" if record["treatment"] == "with" else "control",
            "replicate": f"sample_{record['sample']}", "locked_split": split,
        }, index=unique_ids))
        values = protein.to_numpy(dtype=np.float32).T
        cell_col = np.repeat(np.asarray(unique_ids, dtype=object), len(reagents))
        reagent_col = np.tile(np.asarray(reagents, dtype=object), len(unique_ids))
        long_blocks.append(pd.DataFrame({
            "cell_id": cell_col, "reagent_id": reagent_col,
            "gene_id": [reagent_gene(x) for x in reagent_col], "value": values.ravel(),
        }))

    counts = sparse.vstack(rna_blocks, format="csr", dtype=np.int32)
    protein = np.vstack(protein_blocks).astype(np.float32)
    obs = pd.concat(obs_blocks)
    development = obs["locked_split"].eq("development").to_numpy()
    labels, threshold_metadata = protein_proxy_labels(protein, reagents, development)
    obs["cell_type"] = labels
    obs["cell_type_broad"] = labels
    var = pd.DataFrame(index=pd.Index(selected_genes, name="gene_id"))
    var["feature_name"] = var.index.astype(str)
    var["curated_marker"] = var.index.isin(RNA_MARKERS)
    prepared = ad.AnnData(X=counts, obs=obs, var=var)
    prepared.layers["counts"] = counts.copy()
    prepared.obsm["protein_counts"] = protein
    prepared.uns["protein_reagents"] = {
        "reagent_id": reagents, "gene_id": [reagent_gene(x) for x in reagents],
        "identity_rule": "(cell_id, reagent_id, gene_id); duplicate gene targets remain separate",
    }
    prepared.uns["protein_proxy_labeling"] = threshold_metadata
    prepared.uns["source"] = {
        "accession": "GSE100501", "role": "donor-matched CITE-seq RNA/protein",
        "files_json": json.dumps(
            [{"path": str(record["rna"]), "sha256": sha256_file(record["rna"])} for record in records]
            + [{"path": str(record["protein"]), "sha256": sha256_file(record["protein"])} for record in records],
            sort_keys=True,
        ),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_h5ad(output, compression="gzip")
    protein_long = pd.concat(long_blocks, ignore_index=True)
    if protein_long.duplicated(["cell_id", "reagent_id", "gene_id"]).any():
        raise ValueError("duplicate protein proxy keys after preparation")
    protein_path = Path(args.protein_long).resolve()
    protein_path.parent.mkdir(parents=True, exist_ok=True)
    protein_long.to_parquet(protein_path, index=False)

    report = {
        "accession": "GSE100501", "output": str(output), "output_sha256": sha256_file(output),
        "protein_long": str(protein_path), "protein_long_sha256": sha256_file(protein_path),
        "n_cells": prepared.n_obs, "n_genes": prepared.n_vars, "n_reagents": len(reagents),
        "n_protein_proxy_rows": len(protein_long), "n_donors": int(obs["donor"].nunique()),
        "cells_by_split": obs["locked_split"].value_counts().sort_index().to_dict(),
        "protein_proxy_labels": obs["cell_type"].value_counts().sort_index().to_dict(),
        "raw_integer_counts": bool(np.all(counts.data == np.floor(counts.data))),
        "selection_and_threshold_fit": "development donor 1 only",
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
