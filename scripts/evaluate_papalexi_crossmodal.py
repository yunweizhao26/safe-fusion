#!/usr/bin/env python3
"""Evaluate held-out Safe Fusion zero scores against matched ADT measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


PROTEIN_TO_GENE = {
    "CD86": "CD86",
    "PDL1": "CD274",
    "PDL2": "PDCD1LG2",
    "CD366": "HAVCR2",
}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return float("nan")
    return float(spearmanr(left, right).statistic)


def quantile_metrics(labels: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if np.unique(labels).size < 2 or np.unique(score).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, score)), float(average_precision_score(labels, score))


def stratified_bootstrap_rho(
    frame: pd.DataFrame,
    score_column: str,
    protein_column: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    groups = [value.index.to_numpy() for _, value in frame.groupby("replicate", observed=True)]
    values = []
    for _ in range(iterations):
        sampled = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in groups]
        )
        sample = frame.loc[sampled]
        values.append(safe_spearman(sample[score_column].to_numpy(), sample[protein_column].to_numpy()))
    values = np.asarray(values, dtype=float)
    return float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def grouped_permutation_p(
    frame: pd.DataFrame,
    score_column: str,
    protein_column: str,
    grouping: list[str],
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = safe_spearman(frame[score_column].to_numpy(), frame[protein_column].to_numpy())
    null = np.empty(iterations, dtype=float)
    group_indices = [value.index.to_numpy() for _, value in frame.groupby(grouping, observed=True)]
    protein = frame[protein_column].copy()
    for iteration in range(iterations):
        shuffled = protein.copy()
        for indices in group_indices:
            shuffled.loc[indices] = rng.permutation(protein.loc[indices].to_numpy())
        null[iteration] = safe_spearman(frame[score_column].to_numpy(), shuffled.to_numpy())
    p_value = (1 + int(np.sum(np.abs(null) >= abs(observed)))) / (iterations + 1)
    return float(observed), float(p_value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--panel", required=True)
    parser.add_argument("--selector-scores", required=True)
    parser.add_argument("--fusion-mean", required=True)
    parser.add_argument("--graph-mean", required=True)
    parser.add_argument("--scvi-mean", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--protein-column",
        choices=["adt_clr", "adt_count"],
        default="adt_clr",
    )
    parser.add_argument(
        "--gene",
        nargs="+",
        default=None,
        help="Optionally restrict evaluation to named RNA genes.",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    truth = ad.read_h5ad(args.prepared)
    corrupted = ad.read_h5ad(args.corrupted)
    if not np.array_equal(truth.obs_names.astype(str), corrupted.obs_names.astype(str)):
        raise ValueError("Prepared and corrupted cell orders differ")
    if not np.array_equal(truth.var_names.astype(str), corrupted.var_names.astype(str)):
        raise ValueError("Prepared and corrupted gene orders differ")
    truth_counts = dense(truth.layers["counts"] if "counts" in truth.layers else truth.X)
    corrupted_counts = dense(corrupted.layers["corrupted_counts"])
    library_size = np.log1p(corrupted_counts.sum(axis=1))
    fused = np.load(args.fusion_mean, mmap_mode="r")
    graph = np.load(args.graph_mean, mmap_mode="r")
    scvi = np.load(args.scvi_mean, mmap_mode="r")
    for name, matrix in {"fusion": fused, "graph": graph, "scvi": scvi}.items():
        if matrix.shape != truth.shape:
            raise ValueError(f"{name} matrix shape {matrix.shape} differs from {truth.shape}")

    source_column = "source_cell_id" if "source_cell_id" in truth.obs else None
    if source_column is None:
        raise ValueError("Prepared data does not record source_cell_id")
    cell_meta = pd.DataFrame(
        {
            "cell_id": truth.obs_names.astype(str),
            "source_cell_id": truth.obs[source_column].astype(str).to_numpy(),
        }
    )
    scores = pd.read_parquet(args.selector_scores)
    scores = scores.loc[scores["split"].astype(str) == "test"].copy()
    scores = scores.merge(cell_meta, on="cell_id", validate="many_to_one")
    panel = pd.read_parquet(args.panel)

    threshold_rows: list[dict] = []
    continuous_rows: list[dict] = []
    fill_rows: list[dict] = []
    replicate_rows: list[dict] = []
    permutation_rows: list[dict] = []
    availability_rows: list[dict] = []
    rng = np.random.default_rng(args.seed)
    protein_column = args.protein_column
    quantiles = np.linspace(0.30, 0.70, 41)
    fill_fractions = np.linspace(0.01, 0.20, 100)
    gene_index = {str(gene): index for index, gene in enumerate(truth.var_names)}

    for protein, gene in PROTEIN_TO_GENE.items():
        if args.gene and gene not in set(args.gene):
            continue
        available = gene in gene_index and bool((scores["gene_id"].astype(str) == gene).any())
        availability_rows.append(
            {
                "protein": protein,
                "gene": gene,
                "available_in_prepared_benchmark": available,
                "direct_perturbation_target": gene in {"CD86", "PDCD1LG2"},
            }
        )
        if not available:
            continue
        gi = gene_index[gene]
        subset = scores.loc[scores["gene_id"].astype(str) == gene].copy()
        protein_panel = panel.loc[panel["protein"].astype(str) == protein, [
            "cell_id", "replicate", "target", "adt_count", "adt_clr"
        ]].rename(columns={"cell_id": "source_cell_id"})
        subset = subset.merge(protein_panel, on="source_cell_id", validate="many_to_one")
        ci = subset["cell_index"].to_numpy(dtype=int)
        subset["truth_count"] = truth_counts[ci, gi]
        subset["mlp_safe_fusion"] = subset["selector_score"].astype(float)
        subset["fused_component"] = np.asarray(fused[ci, gi], dtype=float)
        subset["weighted_knn"] = np.asarray(graph[ci, gi], dtype=float)
        subset["scvi"] = np.asarray(scvi[ci, gi], dtype=float)
        subset["library_size"] = library_size[ci]
        subset = subset.loc[
            (subset["truth_count"] == 0) & (subset["masked_positive"].astype(int) == 0)
        ].copy()
        subset.reset_index(drop=True, inplace=True)
        methods = ["mlp_safe_fusion", "scvi", "weighted_knn", "fused_component", "library_size"]
        scopes = [("all_test_rna_zeros", subset)]
        if gene in {"CD86", "PDCD1LG2"}:
            scopes.append(
                ("excluding_direct_target", subset.loc[subset["target"].astype(str) != gene].copy())
            )
        for scope, frame in scopes:
            frame = frame.reset_index(drop=True)
            if len(frame) < 20:
                continue
            analysis_role = (
                "independent_readout"
                if gene in {"CD274", "HAVCR2"}
                else "direct_target_sensitivity"
            )
            for method in methods:
                score = frame[method].to_numpy(dtype=float)
                rho = safe_spearman(score, frame[protein_column].to_numpy(dtype=float))
                ci_low, ci_high = stratified_bootstrap_rho(
                    frame, method, protein_column, args.bootstrap, rng
                )
                continuous_rows.append(
                    {
                        "protein": protein,
                        "gene": gene,
                        "scope": scope,
                        "analysis_role": analysis_role,
                        "method": method,
                        "n_rna_zero_cells": int(len(frame)),
                        "spearman": rho,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
                for replicate, replicate_frame in frame.groupby("replicate", observed=True):
                    replicate_auc = []
                    for quantile in quantiles:
                        replicate_threshold = float(
                            replicate_frame[protein_column].quantile(quantile)
                        )
                        replicate_labels = (
                            replicate_frame[protein_column].to_numpy(dtype=float)
                            > replicate_threshold
                        ).astype(int)
                        replicate_auc.append(
                            quantile_metrics(
                                replicate_labels,
                                replicate_frame[method].to_numpy(dtype=float),
                            )[0]
                        )
                    replicate_rows.append(
                        {
                            "protein": protein,
                            "gene": gene,
                            "scope": scope,
                            "method": method,
                            "replicate": str(replicate),
                            "n_rna_zero_cells": int(len(replicate_frame)),
                            "spearman": safe_spearman(
                                replicate_frame[method].to_numpy(dtype=float),
                                replicate_frame[protein_column].to_numpy(dtype=float),
                            ),
                            "mean_roc_auc_q30_q70": float(np.nanmean(replicate_auc)),
                            "minimum_roc_auc_q30_q70": float(np.nanmin(replicate_auc)),
                            "maximum_roc_auc_q30_q70": float(np.nanmax(replicate_auc)),
                        }
                    )
                for grouping_name, grouping in [
                    ("replicate", ["replicate"]),
                    ("replicate_and_target", ["replicate", "target"]),
                ]:
                    observed, p_value = grouped_permutation_p(
                        frame, method, protein_column, grouping, args.permutations, rng
                    )
                    permutation_rows.append(
                        {
                            "protein": protein,
                            "gene": gene,
                            "scope": scope,
                            "method": method,
                            "permutation_groups": grouping_name,
                            "iterations": args.permutations,
                            "spearman": observed,
                            "two_sided_p": p_value,
                        }
                    )
                for quantile in quantiles:
                    threshold = float(frame[protein_column].quantile(quantile))
                    labels = (frame[protein_column].to_numpy(dtype=float) > threshold).astype(int)
                    roc_auc, pr_auc = quantile_metrics(labels, score)
                    threshold_rows.append(
                        {
                            "protein": protein,
                            "gene": gene,
                            "scope": scope,
                            "method": method,
                            "protein_threshold_quantile": float(quantile),
                            "protein_threshold": threshold,
                            "protein_high_fraction": float(labels.mean()),
                            "roc_auc": roc_auc,
                            "pr_auc": pr_auc,
                        }
                    )
                order = np.argsort(-score, kind="stable")
                background = float(frame[protein_column].mean())
                for fraction in fill_fractions:
                    selected = max(1, int(round(float(fraction) * len(frame))))
                    selected_mean = float(frame[protein_column].to_numpy(dtype=float)[order[:selected]].mean())
                    fill_rows.append(
                        {
                            "protein": protein,
                            "gene": gene,
                            "scope": scope,
                            "method": method,
                            "selected_fraction": float(fraction),
                            "n_selected": selected,
                            "selected_protein_mean": selected_mean,
                            "all_rna_zero_protein_mean": background,
                            "protein_enrichment": selected_mean - background,
                        }
                    )

    availability = pd.DataFrame(availability_rows)
    threshold = pd.DataFrame(threshold_rows)
    continuous = pd.DataFrame(continuous_rows)
    fill = pd.DataFrame(fill_rows)
    replicate = pd.DataFrame(replicate_rows)
    permutation = pd.DataFrame(permutation_rows)
    availability.to_csv(output / "pair_availability.csv", index=False)
    threshold.to_csv(output / "protein_threshold_range_metrics.csv", index=False)
    continuous.to_csv(output / "continuous_protein_association.csv", index=False)
    fill.to_csv(output / "fill_range_protein_enrichment.csv", index=False)
    replicate.to_csv(output / "replicate_association.csv", index=False)
    permutation.to_csv(output / "permutation_tests.csv", index=False)

    summary_rows = []
    for keys, group in threshold.groupby(["protein", "gene", "scope", "method"], observed=True):
        protein, gene, scope, method = keys
        fill_group = fill.loc[
            (fill["protein"] == protein)
            & (fill["gene"] == gene)
            & (fill["scope"] == scope)
            & (fill["method"] == method)
        ]
        continuous_row = continuous.loc[
            (continuous["protein"] == protein)
            & (continuous["gene"] == gene)
            & (continuous["scope"] == scope)
            & (continuous["method"] == method)
        ].iloc[0]
        summary_rows.append(
            {
                "protein": protein,
                "gene": gene,
                "scope": scope,
                "method": method,
                "n_rna_zero_cells": int(continuous_row["n_rna_zero_cells"]),
                "spearman": float(continuous_row["spearman"]),
                "spearman_ci_low": float(continuous_row["ci_low"]),
                "spearman_ci_high": float(continuous_row["ci_high"]),
                "mean_roc_auc_q30_q70": float(group["roc_auc"].mean()),
                "minimum_roc_auc_q30_q70": float(group["roc_auc"].min()),
                "maximum_roc_auc_q30_q70": float(group["roc_auc"].max()),
                "mean_pr_auc_q30_q70": float(group["pr_auc"].mean()),
                "mean_protein_enrichment_fill01_fill20": float(fill_group["protein_enrichment"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "method_summary.csv", index=False)

    lead_rows = []
    for keys, group in threshold.groupby(
        ["protein", "gene", "scope", "protein_threshold_quantile"], observed=True
    ):
        best = group.loc[group["roc_auc"].idxmax()]
        lead_rows.append({
            "comparison": "protein_threshold_range",
            "protein": keys[0],
            "gene": keys[1],
            "scope": keys[2],
            "point": float(keys[3]),
            "best_method": str(best["method"]),
        })
    for keys, group in fill.groupby(
        ["protein", "gene", "scope", "selected_fraction"], observed=True
    ):
        best = group.loc[group["protein_enrichment"].idxmax()]
        lead_rows.append({
            "comparison": "fill_fraction_range",
            "protein": keys[0],
            "gene": keys[1],
            "scope": keys[2],
            "point": float(keys[3]),
            "best_method": str(best["method"]),
        })
    leads = pd.DataFrame(lead_rows)
    leads.to_csv(output / "range_leaders.csv", index=False)

    report = {
        "test_only": True,
        "test_labels_used_for_selector_training": False,
        "protein_column": protein_column,
        "natural_rna_zero_interpretation": (
            "Protein abundance provides an independent matched measurement that can support residual gene expression. "
            "It does not identify the biological cause of each RNA zero."
        ),
        "protein_threshold_quantiles": [float(quantiles.min()), float(quantiles.max()), int(len(quantiles))],
        "selected_fill_fraction_range": [float(fill_fractions.min()), float(fill_fractions.max()), int(len(fill_fractions))],
        "available_pairs": availability.to_dict(orient="records"),
        "range_leader_counts": (
            leads.groupby(["comparison", "protein", "gene", "scope", "best_method"], observed=True)
            .size()
            .rename("points_led")
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
