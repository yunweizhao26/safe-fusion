#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "paper_evidence" / "benchmark_data" / "benchmark_data.csv"

LABELS = {
    "pancreas_0": "Pancreas, fold 1",
    "pancreas_1": "Pancreas, fold 2",
    "pancreas_2": "Pancreas, fold 3",
    "colon": "Colon",
    "norman_crispra": "Norman CRISPRa",
    "adamson_crispri": "Adamson CRISPRi",
    "dixit_ko": "Dixit knockout",
    "papalexi_eccite": "Papalexi ECCITE-seq",
    "zebrafish": "Zebrafish time course",
    "papalexi_crossmodal": "Papalexi RNA-protein",
}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def summarize(key: str, corrupted: Path, coordinates: Path, splits: Path) -> dict:
    adata = ad.read_h5ad(corrupted)
    split = (
        pd.read_parquet(splits)
        .set_index("cell_id")
        .loc[adata.obs_names.astype(str), "split"]
        .to_numpy()
    )
    test_rows = np.flatnonzero(split == "test")
    candidates = dense(adata.layers["corrupted_counts"][test_rows]) == 0

    mask = pd.read_parquet(coordinates, columns=["cell_index", "gene_index"])
    position = np.full(adata.n_obs, -1, dtype=np.int64)
    position[test_rows] = np.arange(len(test_rows))
    test_position = position[mask["cell_index"].to_numpy(dtype=np.int64)]
    in_test = test_position >= 0
    masked = np.zeros_like(candidates)
    masked[test_position[in_test], mask["gene_index"].to_numpy(dtype=np.int64)[in_test]] = True

    n_candidates = int(candidates.sum())
    n_positives = int((candidates & masked).sum())
    return {
        "dataset": key,
        "label": LABELS.get(key, key),
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "test_cells": int(len(test_rows)),
        "test_candidates": n_candidates,
        "masked_positives": n_positives,
        "prevalence_pct": 100.0 * n_positives / n_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--unit",
        nargs=4,
        action="append",
        required=True,
        metavar=("KEY", "CORRUPTED", "COORDINATES", "SPLITS"),
        help="Dataset key, masked input h5ad, mask coordinates parquet and split parquet. Repeat per dataset.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT, help="CSV path for the table")
    args = parser.parse_args()

    rows = []
    for key, corrupted, coordinates, splits in args.unit:
        rows.append(summarize(key, Path(corrupted), Path(coordinates), Path(splits)))
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
