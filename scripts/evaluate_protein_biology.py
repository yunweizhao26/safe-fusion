#!/usr/bin/env python3
"""Evaluate RNA outputs against locked, independently measured CITE-seq proteins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def dense(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(spearmanr(x, y).statistic)


def knn_indices(matrix: np.ndarray, seed: int, k: int = 15) -> np.ndarray:
    n_components = min(30, matrix.shape[0] - 1, matrix.shape[1] - 1)
    transformed = PCA(n_components=n_components, random_state=seed).fit_transform(matrix)
    return NearestNeighbors(n_neighbors=min(k + 1, len(matrix))).fit(transformed).kneighbors(return_distance=False)[:, 1:]


def neighborhood_overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x).intersection(y)) / len(x) for x, y in zip(a, b, strict=True)]))


def load_prediction(contract: Path) -> np.ndarray | None:
    if (contract / "failure.json").exists():
        return None
    return np.load(contract / "mean.npy", mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--contracts-root", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    truth = ad.read_h5ad(args.truth)
    corrupted = ad.read_h5ad(args.corrupted)
    splits = pd.read_parquet(args.splits).set_index("cell_id").loc[truth.obs_names.astype(str)]
    test = splits["split"].eq("test").to_numpy()
    development = splits["split"].eq("development").to_numpy()
    test_indices = np.flatnonzero(test)
    protein = np.asarray(truth.obsm["protein_counts"], dtype=np.float64)
    reagents = list(map(str, truth.uns["protein_reagents"]["reagent_id"]))
    reagent_genes = list(map(str, truth.uns["protein_reagents"]["gene_id"]))
    gene_index = {str(gene): i for i, gene in enumerate(truth.var_names)}
    matched = [(i, gene_index[gene], reagent, gene) for i, (reagent, gene) in enumerate(zip(reagents, reagent_genes, strict=True)) if gene in gene_index]
    if not matched:
        raise ValueError("no RNA/protein gene mappings overlap")

    # Per-reagent high/low protein cutoffs are frozen using development donor 1.
    log_protein = np.log1p(protein)
    low_cut = np.quantile(log_protein[development], 0.25, axis=0)
    high_cut = np.quantile(log_protein[development], 0.75, axis=0)
    protein_neighbors = knn_indices(log_protein[test], args.seed)
    original = dense(truth.layers["counts"])[test].astype(np.float64)
    corrupted_counts = dense(corrupted.layers["corrupted_counts"])[test].astype(np.float64)
    coordinate_frame = pd.read_parquet(args.coordinates, filters=[("split", "==", "test")])
    masked_keys = set(zip(coordinate_frame["cell_index"].astype(int), coordinate_frame["gene_index"].astype(int), strict=True))

    rows: list[dict] = []
    for specification in args.methods:
        if "=" in specification:
            method, explicit_path = specification.split("=", 1)
            contract = Path(explicit_path)
        else:
            method = specification
            contract = Path(args.contracts_root) / method
        prediction = load_prediction(contract)
        if prediction is None:
            rows.append({"method": method, "scope": "overall", "metric": "method_failure", "value": np.nan})
            continue
        predicted = np.asarray(prediction[test], dtype=np.float64)
        rna_neighbors = knn_indices(np.log1p(predicted), args.seed)
        rows.append({"method": method, "scope": "overall", "metric": "protein_rna_knn_overlap", "value": neighborhood_overlap(rna_neighbors, protein_neighbors), "n": int(test.sum())})

        reagent_correlations = []
        high_recovery = []
        low_false_fill = []
        protein_zero_contrasts = []
        for reagent_i, gene_i, reagent, gene in matched:
            protein_values = log_protein[test, reagent_i]
            expression = np.log1p(predicted[:, gene_i])
            rho = correlation(expression, protein_values)
            reagent_correlations.append(rho)
            rows.append({"method": method, "scope": "reagent", "reagent_id": reagent, "gene_id": gene, "metric": "rna_protein_spearman", "value": rho, "n": int(test.sum())})

            originally_zero = original[:, gene_i] == 0
            high = originally_zero & (protein_values >= high_cut[reagent_i])
            low = originally_zero & (protein_values <= low_cut[reagent_i])
            if high.any():
                value = float(np.mean(predicted[high, gene_i] > 0.5))
                high_recovery.append(value)
                rows.append({"method": method, "scope": "reagent", "reagent_id": reagent, "gene_id": gene, "metric": "high_protein_zero_fill_rate", "value": value, "n": int(high.sum())})
            if low.any():
                value = float(np.mean(predicted[low, gene_i] > 0.5))
                low_false_fill.append(value)
                rows.append({"method": method, "scope": "reagent", "reagent_id": reagent, "gene_id": gene, "metric": "low_protein_zero_fill_rate", "value": value, "n": int(low.sum())})
            if high.any() and low.any():
                contrast = float(np.mean(np.log1p(predicted[high, gene_i])) - np.mean(np.log1p(predicted[low, gene_i])))
                protein_zero_contrasts.append(contrast)
                rows.append({"method": method, "scope": "reagent", "reagent_id": reagent, "gene_id": gene, "metric": "high_vs_low_protein_zero_expression_contrast", "value": contrast, "n": int(high.sum() + low.sum())})

            # Masked RNA positives are truth-bearing; protein is only used to
            # stratify them, never to redefine their recovery label.
            local_masked = np.asarray([(global_i, gene_i) in masked_keys for global_i in test_indices])
            if local_masked.any():
                mae = float(np.mean(np.abs(np.log1p(predicted[local_masked, gene_i]) - np.log1p(original[local_masked, gene_i]))))
                rows.append({"method": method, "scope": "reagent", "reagent_id": reagent, "gene_id": gene, "metric": "masked_positive_log1p_mae", "value": mae, "n": int(local_masked.sum())})

        for metric, values in (
            ("median_rna_protein_spearman", reagent_correlations),
            ("median_high_protein_zero_fill_rate", high_recovery),
            ("median_low_protein_zero_fill_rate", low_false_fill),
            ("median_high_vs_low_protein_zero_expression_contrast", protein_zero_contrasts),
        ):
            rows.append({"method": method, "scope": "overall", "metric": metric, "value": float(np.nanmedian(values)), "n": int(np.isfinite(values).sum())})

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    overall = frame[frame["scope"].eq("overall")].pivot(index="method", columns="metric", values="value").reset_index()
    payload = {
        "interpretation": "Protein is an independent continuous proxy, not certified RNA absence or cell identity.",
        "test_unit": sorted(splits.loc[test, "biological_unit"].astype(str).unique().tolist()),
        "n_test_cells": int(test.sum()), "n_reagents": len(reagents), "n_matched_reagents": len(matched),
        "threshold_fit": "development donor 1 only", "overall": overall.where(pd.notnull(overall), None).to_dict(orient="records"),
    }
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
