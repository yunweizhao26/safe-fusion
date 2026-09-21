#!/usr/bin/env python3
"""Fast graph-embedding surrogate of GEARS over imputed inputs.

Uses the same GO graph as GEARS (Norman go.csv), builds a spectral embedding
of the gene graph, and predicts held-out perturbation responses with ridge
regression from that embedding. This is a linearized GEARS-style predictor:
it is fast and directly answers whether imputation changes held-out
perturbation-response prediction, with the official GEARS sweep run
separately on Slurm/GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def log1p_cpm(matrix: np.ndarray) -> np.ndarray:
    library = matrix.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(matrix * scale[:, None]).astype(np.float32)


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 100) -> float:
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    return float(len(top_a & top_b) / k)


def parse_method(value: str) -> tuple[str, Path | None]:
    if value in ("reference_truth", "corrupted_raw"):
        return value, None
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--go-edges", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth_counts = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted_counts = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    targets = np.asarray(truth_adata.obs["target"].astype(str).to_numpy(), dtype=str)
    conditions = np.asarray(truth_adata.obs["condition"].astype(str).to_numpy(), dtype=str)
    controls = conditions == "ctrl"
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        truth_adata.obs_names.astype(str), "split"
    ].to_numpy()
    development = split == "development"
    test = split == "test"
    gene_names = (
        truth_adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in truth_adata.var
        else np.asarray(truth_adata.var_names, dtype=str)
    )
    gene_lookup = {gene: index for index, gene in enumerate(gene_names)}
    condition_levels = sorted(set(targets) - {"none"})

    gene2go = pd.read_pickle(REPOSITORY / "external_data/perturbation_response/gears_aux/gene2go_all.pkl")
    gene_subset = np.asarray([index for index, gene in enumerate(gene_names) if gene in gene2go], dtype=int)
    gene_names = gene_names[gene_subset]
    truth_counts = truth_counts[:, gene_subset]
    corrupted_counts = corrupted_counts[:, gene_subset]
    gene_lookup = {gene: index for index, gene in enumerate(gene_names)}
    condition_levels = [c for c in condition_levels if c in gene_lookup]

    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(round(len(condition_levels) * args.test_fraction)))
    test_conditions = sorted(rng.choice(condition_levels, size=n_test, replace=False).tolist())
    train_conditions = [c for c in condition_levels if c not in test_conditions]
    train_genes = np.asarray([gene_lookup[c] for c in train_conditions], dtype=int)
    test_genes = np.asarray([gene_lookup[c] for c in test_conditions], dtype=int)

    # GO graph adjacency (row-normalized).
    go_edges = pd.read_csv(args.go_edges)
    index = {gene: i for i, gene in enumerate(gene_names)}
    go_edges = go_edges[go_edges["source"].isin(index) & go_edges["target"].isin(index)]
    src = go_edges["source"].map(index).to_numpy(dtype=np.int64)
    dst = go_edges["target"].map(index).to_numpy(dtype=np.int64)
    weights = go_edges["importance"].to_numpy(dtype=np.float64)
    del go_edges
    adjacency = sparse.csr_matrix((weights, (src, dst)), shape=(len(gene_names), len(gene_names)), dtype=np.float64)
    del src, dst, weights
    adjacency = (adjacency + adjacency.T).tocsr()
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    row_sum_inv = np.zeros_like(row_sum)
    valid = row_sum > 0
    row_sum_inv[valid] = 1.0 / row_sum[valid]
    adjacency_norm = sparse.diags(row_sum_inv) @ adjacency
    del adjacency
    print("GO neighbors computed")

    reference_norm = log1p_cpm(truth_counts)
    reference_ctrl_mean = reference_norm[test & controls].mean(axis=0)
    reference_test_responses = np.vstack([
        reference_norm[test & (targets == condition)].mean(axis=0) - reference_ctrl_mean
        for condition in test_conditions
    ]).astype(np.float32)

    methods: list[tuple[str, Path | None]] = []
    for specification in args.method:
        methods.append(parse_method(specification))
    methods.append(("reference_truth", None))
    methods.append(("corrupted_raw", None))

    summary_rows = []
    for name, path in methods:
        if name == "reference_truth":
            matrix = truth_counts
        elif name == "corrupted_raw":
            matrix = corrupted_counts
        else:
            matrix = np.load(path / "mean.npy", allow_pickle=False)[:, gene_subset]
        normalized = log1p_cpm(matrix)
        control_development_mean = normalized[development & controls].mean(axis=0)
        control_development = matrix[development & controls].astype(np.float64)
        own_stats = np.column_stack([
            control_development.mean(axis=0),
            1.0 - (control_development > 0).mean(axis=0),
            control_development.var(axis=0),
        ])
        neighbor_stats = adjacency_norm @ own_stats
        gene_features = np.concatenate([own_stats, neighbor_stats], axis=1).astype(np.float32)
        del control_development, own_stats, neighbor_stats
        train_response = np.vstack([
            normalized[development & (targets == condition)].mean(axis=0) - control_development_mean
            for condition in train_conditions
        ]).astype(np.float32)
        model = Ridge(alpha=args.alpha)
        model.fit(gene_features[train_genes], train_response)
        predicted = model.predict(gene_features[test_genes])
        per_condition = []
        for index, condition in enumerate(test_conditions):
            pred_response = predicted[index]
            ref_response = reference_test_responses[index]
            valid = np.isfinite(pred_response) & np.isfinite(ref_response)
            pearson = float(pearsonr(pred_response[valid], ref_response[valid])[0]) if valid.sum() > 2 else float("nan")
            overlap = top_k_overlap(pred_response, ref_response)
            rmse = float(np.sqrt(np.mean((pred_response[valid] - ref_response[valid]) ** 2)))
            per_condition.append({"method": name, "condition": condition, "pearson": pearson, "top100_overlap": overlap, "rmse": rmse})
        frame = pd.DataFrame(per_condition)
        row = {
            "method": name,
            "n_test_conditions": int(len(frame)),
            "mean_pearson": float(frame["pearson"].mean()),
            "mean_top100_overlap": float(frame["top100_overlap"].mean()),
            "mean_rmse": float(frame["rmse"].mean()),
        }
        summary_rows.append(row)
        print(json.dumps(row))

    summary = pd.DataFrame(summary_rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output / "summary.parquet", index=False)
    report = {
        "design": "GO spectral embedding + ridge; held-out perturbation response prediction; reference = uncorrupted truth on test cells",
        "embedding_dim": 0,
        "alpha": args.alpha,
        "seed": args.seed,
        "test_conditions": test_conditions,
        "results": summary.to_dict(orient="records"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
