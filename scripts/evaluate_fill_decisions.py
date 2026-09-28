#!/usr/bin/env python3
"""Fill decisions of standard imputers on recorded zeros (Supplementary Table S1).

Reads the recorded matrix ``X`` of a CELLxGENE H5AD and the imputer outputs
written by ``scripts/standard_imputers/run_standard_imputers.py`` under
``<imputed-root>/<method>/<dataset-id>/<disease>/<tissue>.npy``. Cells are
restricted to ``is_primary_data`` as in the imputer runs. A recorded zero is
filled by a method when the method's output at that entry is positive. MAGIC
returns only the genes detected in at least ``--magic-min-cells`` cells of the
subset (its modeled genes), and its other genes count as not filled.

Outputs in ``--output-dir``:

* ``fill_rates.csv``: per subset, method and gene set (all genes or MAGIC
  modeled genes), the recorded zeros, the filled zeros, the fill rate, and the
  fraction of all output entries of those genes that are positive.
* ``gene_patterns.csv``: per subset, method and gene set, the fractions of
  genes with recorded zeros whose zeros are all kept, all filled or mixed, and
  the share of recorded zeros that lie in mixed genes.
* ``pair_agreement.csv``: per subset and method pair, the observed fraction of
  recorded zeros on which the two methods make the same decision, and the
  agreement expected if the methods decided independently, either at their
  overall fill rates or at their fill rates within each gene.
* ``table_s1.csv``: fill rates in percent, one row per method and one column
  per subset, with the modeled-genes row of MAGIC.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
# Output directory name and display name of each method.
METHODS = {
    "SAUCIE": "SAUCIE",
    "MAGIC": "MAGIC",
    "deepImpute": "DeepImpute",
    "scScope": "scScope",
    "scVI": "scVI",
    "knn_smoothing": "kNN smoothing",
}
MAGIC = "MAGIC"
SUBSETS = (
    "Crohn disease|caecum",
    "Crohn disease|caecum epithelium",
    "Crohn disease|right colon",
    "Crohn disease|lamina propria of mucosa of colon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="CELLxGENE H5AD that the imputers were run on")
    parser.add_argument(
        "--imputed-root", default=str(REPO / "artifacts/paper_evidence/standard_imputers"),
        help="root of the <method>/<dataset-id>/<disease>/<tissue>.npy outputs",
    )
    parser.add_argument("--dataset-id", help="dataset directory name under each method (default: input file stem)")
    parser.add_argument(
        "--subset", action="append",
        help="'disease|tissue' subset; repeat for several (default: the four subsets of Table S1)",
    )
    parser.add_argument("--magic-min-cells", type=int, default=5, help="MAGIC gene filter of the imputer run")
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--output-dir", default=str(REPO / "artifacts/paper_evidence/fill_decisions"))
    return parser.parse_args()


def subset_rows(obs: pd.DataFrame, disease: str, tissue: str) -> np.ndarray:
    keep = (
        obs["is_primary_data"].to_numpy(dtype=bool)
        & (obs["disease"].astype(str).to_numpy() == disease)
        & (obs["tissue"].astype(str).to_numpy() == tissue)
    )
    return np.flatnonzero(keep)


def dense_rows(adata: ad.AnnData, rows: np.ndarray) -> np.ndarray:
    block = adata.X[rows]
    return block.toarray() if hasattr(block, "toarray") else np.asarray(block)


def count_subset(adata, rows, arrays, magic_min_cells, chunk_rows):
    """Per-gene counts of recorded zeros, fills, joint fills and positive outputs."""
    n_genes = adata.n_vars
    detected = np.zeros(n_genes, dtype=np.int64)
    for start in range(0, len(rows), chunk_rows):
        detected += (dense_rows(adata, rows[start:start + chunk_rows]) != 0).sum(axis=0)
    modeled = np.flatnonzero(detected >= magic_min_cells)
    if arrays[MAGIC].shape[1] != len(modeled):
        raise ValueError(
            f"MAGIC output has {arrays[MAGIC].shape[1]} genes, but {len(modeled)} genes are detected "
            f"in at least {magic_min_cells} cells"
        )
    for name, array in arrays.items():
        if array.shape[0] != len(rows):
            raise ValueError(f"{name} output has {array.shape[0]} rows for {len(rows)} cells")
        if name != MAGIC and array.shape[1] != n_genes:
            raise ValueError(f"{name} output has {array.shape[1]} genes, the input has {n_genes}")

    names = list(arrays)
    pairs = list(combinations(range(len(names)), 2))
    zeros = np.zeros(n_genes, dtype=np.int64)
    filled = np.zeros((len(names), n_genes), dtype=np.int64)
    positive = np.zeros((len(names), n_genes), dtype=np.int64)
    both = np.zeros((len(pairs), n_genes), dtype=np.int64)
    for start in range(0, len(rows), chunk_rows):
        stop = min(start + chunk_rows, len(rows))
        recorded_zero = dense_rows(adata, rows[start:stop]) == 0
        fills = []
        for k, name in enumerate(names):
            chunk = np.asarray(arrays[name][start:stop])
            if name == MAGIC:
                full = np.zeros((stop - start, n_genes), dtype=chunk.dtype)
                full[:, modeled] = chunk
                chunk = full
            is_positive = chunk > 0
            fills.append(is_positive & recorded_zero)
            positive[k] += is_positive.sum(axis=0)
            filled[k] += fills[-1].sum(axis=0)
        zeros += recorded_zero.sum(axis=0)
        for p, (i, j) in enumerate(pairs):
            both[p] += (fills[i] & fills[j]).sum(axis=0)
    in_modeled = np.zeros(n_genes, dtype=bool)
    in_modeled[modeled] = True
    return names, pairs, in_modeled, zeros, filled, positive, both


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    dataset_id = args.dataset_id or input_path.stem
    subsets = args.subset or list(SUBSETS)
    adata = ad.read_h5ad(input_path, backed="r")
    obs = adata.obs

    fill_rows, pattern_rows, pair_rows = [], [], []
    for subset in subsets:
        disease, _, tissue = subset.partition("|")
        rows = subset_rows(obs, disease, tissue)
        arrays = {
            name: np.load(Path(args.imputed_root) / name / dataset_id / disease / f"{tissue}.npy", mmap_mode="r")
            for name in METHODS
        }
        names, pairs, in_modeled, zeros, filled, positive, both = count_subset(
            adata, rows, arrays, args.magic_min_cells, args.chunk_rows,
        )
        n_cells = len(rows)
        gene_sets = {"all genes": np.ones_like(in_modeled), "MAGIC modeled genes": in_modeled}
        for gene_set, genes in gene_sets.items():
            has_zero = genes & (zeros > 0)
            for k, name in enumerate(names):
                fill_rows.append({
                    "subset": subset, "cells": n_cells, "method": METHODS[name], "gene_set": gene_set,
                    "genes": int(genes.sum()), "recorded_zeros": int(zeros[genes].sum()),
                    "filled_zeros": int(filled[k, genes].sum()),
                    "fill_rate": filled[k, genes].sum() / zeros[genes].sum(),
                    "positive_output_fraction": positive[k, genes].sum() / (n_cells * genes.sum()),
                })
                rate = filled[k, has_zero] / zeros[has_zero]
                mixed = (rate > 0) & (rate < 1)
                pattern_rows.append({
                    "subset": subset, "method": METHODS[name], "gene_set": gene_set,
                    "genes_with_zeros": int(has_zero.sum()),
                    "all_kept": np.mean(rate == 0), "all_filled": np.mean(rate == 1), "mixed": np.mean(mixed),
                    "mixed_genes": int(mixed.sum()),
                    "zeros_in_mixed_genes": zeros[has_zero][mixed].sum() / zeros[has_zero].sum(),
                })

        total = zeros.sum()
        gene_rate = filled / np.maximum(zeros, 1)
        for p, (i, j) in enumerate(pairs):
            fill_i, fill_j, joint = filled[i].sum(), filled[j].sum(), both[p].sum()
            observed = (total - fill_i - fill_j + 2 * joint) / total
            rate_i, rate_j = fill_i / total, fill_j / total
            overall_null = rate_i * rate_j + (1 - rate_i) * (1 - rate_j)
            gene_null = (
                zeros * (gene_rate[i] * gene_rate[j] + (1 - gene_rate[i]) * (1 - gene_rate[j]))
            ).sum() / total
            pair_rows.append({
                "subset": subset, "method_a": METHODS[names[i]], "method_b": METHODS[names[j]],
                "observed_agreement": observed, "expected_overall_rates": overall_null,
                "expected_gene_rates": gene_null, "observed_minus_gene_expectation": observed - gene_null,
            })
        print(f"{subset}: {n_cells} cells, {int(in_modeled.sum())} MAGIC modeled genes, {total} recorded zeros",
              flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fill_rates = pd.DataFrame(fill_rows)
    fill_rates.to_csv(output_dir / "fill_rates.csv", index=False)
    pd.DataFrame(pattern_rows).to_csv(output_dir / "gene_patterns.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(output_dir / "pair_agreement.csv", index=False)

    table = fill_rates[fill_rates["gene_set"] == "all genes"].copy()
    modeled = fill_rates[(fill_rates["gene_set"] == "MAGIC modeled genes") & (fill_rates["method"] == METHODS[MAGIC])]
    table = pd.concat([table, modeled.assign(method=f"{METHODS[MAGIC]}, modeled genes")])
    table["column"] = table["subset"].str.partition("|")[2] + " (" + table["cells"].map("{:,}".format) + ")"
    order = list(METHODS.values())
    order.insert(order.index(METHODS[MAGIC]) + 1, f"{METHODS[MAGIC]}, modeled genes")
    wide = table.pivot(index="method", columns="column", values="fill_rate").loc[order]
    wide = wide[table.drop_duplicates("subset")["column"].tolist()]
    (100 * wide).round(1).to_csv(output_dir / "table_s1.csv")
    print((100 * wide).round(1).to_string())


if __name__ == "__main__":
    main()
