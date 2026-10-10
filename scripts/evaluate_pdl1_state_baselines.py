#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from evaluate_papalexi_crossmodal import dense, safe_spearman

PROTEIN, GENE = "PDL1", "CD274"
IFNG_RESPONSE_GENES = (
    "STAT1", "IRF1", "GBP1", "GBP2", "GBP4", "GBP5", "CXCL9", "CXCL10", "CXCL11", "IDO1", "TAP1", "PSMB8",
    "PSMB9", "WARS", "SOCS1", "HLA-DRA", "B2M", "HLA-A", "HLA-B", "HLA-C", "IFITM1", "ISG15", "APOL6",
)
VALUE_RANKINGS = {"fused_value": "safe_fusion", "weighted_knn": "graph_smooth", "svd": "svd_impute", "scvi": "scvi"}
RANKINGS = ("safe_fusion", "svd", "weighted_knn", "scvi", "fused_value", "library_size", "target_baseline", "ifng_score")
QUANTILES = np.linspace(0.30, 0.70, 41)


def mean_threshold_auroc(score: np.ndarray, protein: np.ndarray) -> float:
    ranks = rankdata(score)
    aurocs = []
    for quantile in QUANTILES:
        high = protein > np.quantile(protein, quantile)
        n_high = int(high.sum())
        n_low = len(high) - n_high
        if n_high and n_low:
            aurocs.append((ranks[high].sum() - n_high * (n_high + 1) / 2) / (n_high * n_low))
    return float(np.mean(aurocs))


def partial_spearman(y: np.ndarray, x: np.ndarray, covariate: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), rankdata(covariate)])
    residuals = []
    for values in (rankdata(y), rankdata(x)):
        residuals.append(values - design @ np.linalg.lstsq(design, values, rcond=None)[0])
    return float(np.corrcoef(residuals[0], residuals[1])[0, 1])


def pdl1_frame(prepared: Path, benchmark: Path, panel_path: Path) -> pd.DataFrame:
    truth = ad.read_h5ad(prepared)
    corrupted = ad.read_h5ad(benchmark / "corrupted.h5ad")
    if not np.array_equal(truth.obs_names.astype(str), corrupted.obs_names.astype(str)):
        raise ValueError("Prepared and corrupted cell orders differ")
    if not np.array_equal(truth.var_names.astype(str), corrupted.var_names.astype(str)):
        raise ValueError("Prepared and corrupted gene orders differ")
    gene_index = {str(gene): index for index, gene in enumerate(truth.var_names)}
    missing = [gene for gene in IFNG_RESPONSE_GENES if gene not in gene_index]
    if missing:
        raise ValueError(f"Interferon-gamma response genes missing from the panel: {missing}")
    counts = dense(corrupted.layers["corrupted_counts"])
    library = counts.sum(axis=1)
    normalized = np.log1p(counts / np.maximum(library, 1)[:, None] * 1e4)
    ifng_score = normalized[:, [gene_index[gene] for gene in IFNG_RESPONSE_GENES]].mean(axis=1)

    cells = pd.DataFrame({
        "cell_id": truth.obs_names.astype(str),
        "source_cell_id": truth.obs["source_cell_id"].astype(str).to_numpy(),
    })
    split = pd.read_parquet(benchmark / "splits.parquet").set_index("cell_id").loc[cells["cell_id"], "split"].to_numpy()
    panel = pd.read_parquet(panel_path)
    panel = panel.loc[panel["protein"].astype(str) == PROTEIN, ["cell_id", "replicate", "target", "adt_clr"]]
    panel = panel.rename(columns={"cell_id": "source_cell_id"})
    development = set(cells.loc[split == "development", "source_cell_id"])
    target_means = panel.loc[panel["source_cell_id"].isin(development)].groupby("target")["adt_clr"].mean()

    scores = pd.read_parquet(benchmark / "mlp_selector" / "selected_gene_scores.parquet")
    frame = scores.loc[(scores["split"].astype(str) == "test") & (scores["gene_id"].astype(str) == GENE)]
    frame = frame.merge(cells, on="cell_id", validate="many_to_one")
    frame = frame.merge(panel, on="source_cell_id", validate="many_to_one")
    rows, column = frame["cell_index"].to_numpy(dtype=int), gene_index[GENE]
    truth_counts = truth.layers["counts"] if "counts" in truth.layers else truth.X
    frame["truth_count"] = dense(truth_counts[rows][:, column]).ravel()
    frame["safe_fusion"] = frame["selector_score"].astype(float)
    for name, contract in VALUE_RANKINGS.items():
        frame[name] = np.asarray(np.load(benchmark / contract / "mean.npy", mmap_mode="r")[rows, column], dtype=float)
    frame["library_size"] = np.log1p(library[rows])
    frame["target_baseline"] = frame["target"].map(target_means).astype(float)
    frame["ifng_score"] = ifng_score[rows]
    frame["pdl1_clr"] = frame["adt_clr"].astype(float)
    if frame["target_baseline"].isna().any():
        raise ValueError("A held-out perturbation target has no development cells")
    recorded_zero = (frame["truth_count"] == 0) & (frame["masked_positive"].astype(int) == 0)
    return frame.loc[recorded_zero].reset_index(drop=True)


def statistics(frame: pd.DataFrame, index: np.ndarray) -> dict[str, float]:
    protein = frame["pdl1_clr"].to_numpy()[index]
    values = {}
    for ranking in RANKINGS:
        score = frame[ranking].to_numpy()[index]
        values[f"spearman:{ranking}"] = safe_spearman(score, protein)
        values[f"mean_auroc:{ranking}"] = mean_threshold_auroc(score, protein)
    safe_fusion, svd = frame["safe_fusion"].to_numpy()[index], frame["svd"].to_numpy()[index]
    values["partial:safe_fusion_given_svd"] = partial_spearman(protein, safe_fusion, svd)
    values["partial:svd_given_safe_fusion"] = partial_spearman(protein, svd, safe_fusion)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("external_data/prepared/papalexi_eccite_crossmodal.h5ad"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("artifacts/paper_evidence/papalexi_crossmodal/benchmark"))
    parser.add_argument("--panel", type=Path, default=Path("artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_evidence/papalexi_crossmodal/benchmark/evaluation_pdl1_state"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    frame = pdl1_frame(args.prepared, args.benchmark_root, args.panel)
    point = statistics(frame, np.arange(len(frame)))
    rng = np.random.default_rng(args.seed)
    replicates = [group.index.to_numpy() for _, group in frame.groupby("replicate", observed=True)]
    targets = frame["target"].astype(str).to_numpy()
    target_cells = {target: np.flatnonzero(targets == target) for target in np.unique(targets)}
    schemes = {
        "cells_within_replicate": lambda: np.concatenate([rng.choice(cells, len(cells), replace=True) for cells in replicates]),
        "target_clusters": lambda: np.concatenate(
            [target_cells[target] for target in rng.choice(list(target_cells), len(target_cells), replace=True)]
        ),
    }
    draws = {scheme: pd.DataFrame([statistics(frame, draw()) for _ in range(args.bootstrap)]) for scheme, draw in schemes.items()}

    rankings = pd.DataFrame([
        {"ranking": ranking, "n_cells": len(frame), "spearman": point[f"spearman:{ranking}"],
         "mean_roc_auc_q30_q70": point[f"mean_auroc:{ranking}"]}
        for ranking in RANKINGS
    ])
    paired, partial = [], []
    for scheme, sample in draws.items():
        for comparator in RANKINGS[1:]:
            for metric in ("spearman", "mean_auroc"):
                difference = sample[f"{metric}:safe_fusion"] - sample[f"{metric}:{comparator}"]
                paired.append({
                    "resampling": scheme, "comparator": comparator, "metric": metric,
                    "safe_fusion_minus_comparator": point[f"{metric}:safe_fusion"] - point[f"{metric}:{comparator}"],
                    "ci_low": float(difference.quantile(0.025)), "ci_high": float(difference.quantile(0.975)),
                })
        for name in ("safe_fusion_given_svd", "svd_given_safe_fusion"):
            values = sample[f"partial:{name}"]
            partial.append({
                "resampling": scheme, "partial_correlation": name, "estimate": point[f"partial:{name}"],
                "ci_low": float(values.quantile(0.025)), "ci_high": float(values.quantile(0.975)),
            })
    paired, partial = pd.DataFrame(paired), pd.DataFrame(partial)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(args.output_dir / "ranking_summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_differences.csv", index=False)
    partial.to_csv(args.output_dir / "partial_correlations.csv", index=False)
    report = {
        "protein": PROTEIN,
        "gene": GENE,
        "n_cells": int(len(frame)),
        "n_targets": len(target_cells),
        "ifng_response_genes": list(IFNG_RESPONSE_GENES),
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "target_baseline": "mean development-cell centered log ratio PD-L1 of the cell's perturbation target",
        "ifng_score": "mean log1p count per ten thousand of the response genes in the masked RNA",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(rankings.round(4).to_string(index=False))
    print(paired.round(4).to_string(index=False))
    print(partial.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
