#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.corruption import _quantile_bins

EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"
PANCREAS_RUN = REPOSITORY / "artifacts" / "pancreas_runs" / "0b2469810675-45c81b160d78" / "data" / "pancreas_islets"
COLON_RUN = REPOSITORY / "artifacts" / "colon_runs" / "0b2469810675-c0db6f963e94" / "data" / "colon_epithelial"
NORMAN_PREPARED = REPOSITORY / "external_data" / "prepared" / "norman_crispra.h5ad"

TABLE_ROWS = (
    ("SVD", "SVD"), ("Weighted kNN", "Weighted kNN"), ("ALRA", "ALRA"), ("SAVER", "SAVER"),
    ("MAGIC", "MAGIC"), ("scVI", "scVI"), ("scGPT", "scGPT"),
    ("Selector on SVD", "SVD (stacked)"), ("Selector on MAGIC", "MAGIC (stacked)"),
    ("Selector on scVI", "scVI (stacked)"),
)
DATASETS = ("Pancreas", "Colon", "CRISPRa")
STRATA_BINS = 4
MASK_FRACTION = 0.10

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def table1(production: Path, rebuilt: Path) -> pd.DataFrame:
    rows = []
    for label, root in (("production", production), ("leakage_free", rebuilt)):
        boot = pd.read_csv(root / "masked_f1_paired_bootstrap.csv")
        boot = boot.loc[boot["statistic"] == "mean_difference_1_to_10"].set_index(["dataset", "comparator"])
        summary = json.loads((root / "selector_f1_fillrate_mlp_baselines_summary.json").read_text())
        f1 = json.loads((root / "masked_f1_paired_bootstrap.json").read_text())
        for dataset in DATASETS:
            for row, comparator in TABLE_ROWS:
                value = boot.loc[(dataset, comparator)]
                rows.append({"benchmark": label, "dataset": dataset, "row": row, "estimate_pp": value["estimate_pp"],
                             "lower_pp": value["lower_pp"], "upper_pp": value["upper_pp"], "n_units": int(value["n_units"])})
            share = summary[dataset]["main"]["0.001_0.100"]["fraction_mlp_best"]
            rows.append({"benchmark": label, "dataset": dataset, "row": "Fractions with highest F1",
                         "estimate_pp": 100.0 * share, "n_units": int(f1[dataset]["n_units"])})
            rows.append({"benchmark": label, "dataset": dataset, "row": "Safe Fusion mean F1, 1% to 10%",
                         "estimate_pp": float(np.mean(list(f1[dataset]["safe_fusion_f1_percent"].values()))),
                         "n_units": int(f1[dataset]["n_units"])})
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index=["dataset", "row"], columns="benchmark",
                            values=["estimate_pp", "lower_pp", "upper_pp", "n_units"], aggfunc="first")
    wide.columns = [f"{benchmark}_{value}" for value, benchmark in wide.columns]
    order = {row: index for index, row in enumerate([row for row, _ in TABLE_ROWS] + ["Fractions with highest F1", "Safe Fusion mean F1, 1% to 10%"])}
    wide = wide.reset_index()
    wide["dataset"] = pd.Categorical(wide["dataset"], DATASETS, ordered=True)
    wide = wide.sort_values(["dataset", "row"], key=lambda column: column.map(order) if column.name == "row" else column)
    wide["change_pp"] = wide["leakage_free_estimate_pp"] - wide["production_estimate_pp"]
    return wide.reset_index(drop=True)

def training_bins(values: np.ndarray, training: np.ndarray) -> np.ndarray:

    reference = np.sort(values[training])
    below = np.searchsorted(reference, values, side="left")
    equal = np.searchsorted(reference, values, side="right") - below
    percentile = (below + (equal + 1) / 2) / len(reference)
    return np.minimum((percentile * STRATA_BINS).astype(int), STRATA_BINS - 1)

def strata_change(counts: np.ndarray, training: np.ndarray) -> dict:

    library = counts.sum(axis=1)
    library_all = _quantile_bins(library, STRATA_BINS)
    library_training = training_bins(library, training)
    gene_all = _quantile_bins(counts.mean(axis=0), STRATA_BINS)
    gene_training = _quantile_bins(counts[training].mean(axis=0), STRATA_BINS)
    rows, cols = np.nonzero(counts)
    stratum_all = library_all[rows] * STRATA_BINS + gene_all[cols]
    stratum_training = library_training[rows] * STRATA_BINS + gene_training[cols]
    sizes_all = np.bincount(stratum_all, minlength=STRATA_BINS**2)
    sizes_training = np.bincount(stratum_training, minlength=STRATA_BINS**2)

    def masked(sizes: np.ndarray) -> int:
        return int(sum(max(1, int(round(size * MASK_FRACTION))) for size in sizes if size))

    return {
        "cells": int(counts.shape[0]),
        "training_cells": int(training.sum()),
        "genes": int(counts.shape[1]),
        "cells_with_new_library_quartile": int(np.sum(library_all != library_training)),
        "test_cells_with_new_library_quartile": int(np.sum((library_all != library_training) & ~training)),
        "genes_with_new_mean_quartile": int(np.sum(gene_all != gene_training)),
        "nonzero_entries": int(len(rows)),
        "nonzero_entries_with_new_stratum": int(np.sum(stratum_all != stratum_training)),
        "fraction_nonzero_entries_with_new_stratum": float(np.mean(stratum_all != stratum_training)),
        "masked_entries_all_cell_strata": masked(sizes_all),
        "masked_entries_training_strata": masked(sizes_training),
        "masking_probability_per_stratum": MASK_FRACTION,
    }

def split_of(adata: ad.AnnData, path: Path) -> np.ndarray:
    return pd.read_parquet(path).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()

def target_activation(prepared: Path) -> dict:

    adata = ad.read_h5ad(prepared)
    counts = dense(adata.layers["counts"])
    genes = {gene: index for index, gene in enumerate(adata.var["feature_name"].astype(str))}
    condition = adata.obs["condition"].astype(str).to_numpy()
    control = condition == "ctrl"
    values = []
    for target in sorted(set(adata.obs["target"].astype(str)) - {"none"}):
        gene = genes[target]
        labelled = adata.obs["target"].astype(str).to_numpy() == target
        values.append(float(np.log2(counts[labelled, gene].mean() + 1) - np.log2(counts[control, gene].mean() + 1)))
    values = np.asarray(values)
    return {"targets": int(len(values)), "targets_with_log2fc_at_least_0.5": int(np.sum(values >= 0.5)),
            "median_log2fc": float(np.median(values))}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=EVIDENCE / "review_round2" / "leakage_free")
    parser.add_argument("--production-root", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPOSITORY / args.root

    table = table1(args.production_root, root)
    table.to_csv(root / "table1_old_vs_new.csv", index=False, float_format="%.4f")

    audit: dict[str, object] = {}
    production_pancreas = ad.read_h5ad(PANCREAS_RUN / "preprocessed.h5ad")
    production_genes = set(production_pancreas.var_names.astype(str))
    production_counts = dense(production_pancreas.layers["counts"]).astype(np.float64)
    genes, strata = {}, {}
    for fold in range(3):
        fold_root = root / "pancreas_crossfit" / f"fold_{fold}"
        rebuilt = ad.read_h5ad(fold_root / "prepared.h5ad")
        fold_genes = set(rebuilt.var_names.astype(str))
        genes[f"fold_{fold}"] = {
            "genes": len(fold_genes),
            "curated_markers": int(rebuilt.var["curated_marker"].sum()),
            "shared_with_production": len(fold_genes & production_genes),
            "only_in_fold": len(fold_genes - production_genes),
            "only_in_production": len(production_genes - fold_genes),
            "jaccard": len(fold_genes & production_genes) / len(fold_genes | production_genes),
        }
        splits = fold_root / "splits.parquet"
        strata[f"pancreas_production_fold_{fold}"] = strata_change(
            production_counts, split_of(production_pancreas, splits) != "test")
        strata[f"pancreas_leakage_free_fold_{fold}"] = strata_change(
            dense(rebuilt.layers["counts"]).astype(np.float64), split_of(rebuilt, splits) != "test")
    audit["pancreas_genes"] = {"production_genes": len(production_genes), "folds": genes}

    colon = ad.read_h5ad(COLON_RUN / "preprocessed.h5ad")
    strata["colon"] = strata_change(dense(colon.layers["counts"]).astype(np.float64),
                                    split_of(colon, COLON_RUN / "splits.parquet") != "test")
    production_norman = ad.read_h5ad(NORMAN_PREPARED)
    strata["norman_production"] = strata_change(
        dense(production_norman.layers["counts"]).astype(np.float64),
        split_of(production_norman, EVIDENCE / "norman_crispra" / "splits.parquet") != "test")
    rebuilt_norman = ad.read_h5ad(root / "norman_crispra" / "prepared.h5ad")
    strata["norman_leakage_free"] = strata_change(
        dense(rebuilt_norman.layers["counts"]).astype(np.float64),
        split_of(rebuilt_norman, root / "norman_crispra" / "splits.parquet") != "test")
    audit["mask_strata_training_cells_only"] = strata

    production_report = json.loads(NORMAN_PREPARED.with_suffix(".report.json").read_text())
    rebuilt_report = json.loads((root / "norman_crispra" / "prepared.report.json").read_text())
    qc = pd.DataFrame(rebuilt_report["qc_all_conditions"])
    production_set = {row["condition"] for row in production_report["conditions"]}
    rebuilt_set = {row["condition"] for row in rebuilt_report["conditions"]}
    audit["norman_conditions"] = {
        "single_gene_conditions": rebuilt_report["single_gene_conditions_total"],
        "target_in_gene_list": int(len(qc)),
        "too_few_cells": int(np.sum(qc["reason"] == "too_few_cells")),
        "tested": int(np.sum(qc["reason"] != "too_few_cells")),
        "passed_on_all_cells": len(production_set),
        "passed_on_development_cells": len(rebuilt_set),
        "passed_on_both": len(production_set & rebuilt_set),
        "only_all_cells": sorted(production_set - rebuilt_set),
        "only_development_cells": sorted(rebuilt_set - production_set),
    }
    audit["norman_target_activation_by_label"] = {
        "production": target_activation(NORMAN_PREPARED),
        "leakage_free": target_activation(root / "norman_crispra" / "prepared.h5ad"),
    }
    (root / "leakage_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(json.dumps(audit, indent=2))

if __name__ == "__main__":
    main()
