#!/usr/bin/env python3
"""Perturbation-response prediction: simple linear baseline over imputed inputs.

For each input matrix (reference truth, corrupted raw, graph smoothing, dense
Safe Fusion, calibrated selective), a multi-output ridge model predicts
held-out perturbation response vectors from target-gene features (control
mean expression, dropout rate, variance). Training responses come from
development cells of that input; the evaluation reference is always the
uncorrupted truth on test cells. Metrics are per held-out condition: Pearson
correlation, top-100 overlap, and RMSE between predicted and reference
response vectors.
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
from sklearn.linear_model import Ridge


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.metrics import spearman  # noqa: E402


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def log1p_cpm(matrix: np.ndarray) -> np.ndarray:
    library = matrix.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(matrix * scale[:, None]).astype(np.float32)


def parse_method(value: str) -> tuple[str, Path | None]:
    if value in ("reference_truth", "corrupted_raw"):
        return value, None
    name, path = value.split("=", 1)
    return name, Path(path)


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 100) -> float:
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    return float(len(top_a & top_b) / k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", required=True, help="name=contract_directory, or reference_truth/corrupted_raw")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
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
    rng = np.random.default_rng(args.seed)
    n_test = max(1, int(round(len(condition_levels) * args.test_fraction)))
    test_conditions = sorted(rng.choice(condition_levels, size=n_test, replace=False).tolist())
    train_conditions = [c for c in condition_levels if c not in test_conditions]
    train_genes = np.asarray([gene_lookup[c] for c in train_conditions], dtype=int)
    test_genes = np.asarray([gene_lookup[c] for c in test_conditions], dtype=int)
    print("train conditions:", len(train_conditions), "test conditions:", len(test_conditions))

    methods: list[tuple[str, Path | None]] = []
    for specification in args.method:
        name, path = parse_method(specification)
        methods.append((name, path))
    methods.append(("reference_truth", None))
    methods.append(("corrupted_raw", None))

    # Reference responses on test cells (uncorrupted truth).
    reference_norm = log1p_cpm(truth_counts)
    reference_test_responses = np.vstack([
        reference_norm[test & (targets == condition)].mean(axis=0) - reference_norm[test & controls].mean(axis=0)
        for condition in test_conditions
    ]).astype(np.float32)

    def target_features(matrix: np.ndarray, gene_indices: np.ndarray) -> np.ndarray:
        control_development = matrix[development & controls]
        mean = control_development.mean(axis=0)
        dropout = 1.0 - (control_development > 0).mean(axis=0)
        variance = control_development.var(axis=0)
        selected = np.column_stack([np.ones(len(gene_indices)), mean[gene_indices], dropout[gene_indices], variance[gene_indices]])
        return selected.astype(np.float32)

    def train_responses(matrix: np.ndarray) -> np.ndarray:
        normalized = log1p_cpm(matrix)
        control_development_mean = normalized[development & controls].mean(axis=0)
        return np.vstack([
            normalized[development & (targets == condition)].mean(axis=0) - control_development_mean
            for condition in train_conditions
        ]).astype(np.float32)

    records = []
    for name, path in methods:
        if name == "reference_truth":
            matrix = truth_counts
        elif name == "corrupted_raw":
            matrix = corrupted_counts
        else:
            matrix = np.load(path / "mean.npy", allow_pickle=False)
        train_response = train_responses(matrix)
        train_feature = target_features(matrix, train_genes)
        test_feature = target_features(matrix, test_genes)
        model = Ridge(alpha=args.alpha)
        model.fit(train_feature, train_response)
        predicted = model.predict(test_feature)
        per_condition = []
        for index, condition in enumerate(test_conditions):
            pred = predicted[index]
            ref = reference_test_responses[index]
            valid = np.isfinite(pred) & np.isfinite(ref)
            pearson = float(spearman(ref[valid], pred[valid])) if valid.sum() > 2 else float("nan")
            overlap = top_k_overlap(pred, ref)
            rmse = float(np.sqrt(np.mean((pred - ref) ** 2)))
            per_condition.append({"method": name, "condition": condition, "pearson": pearson, "top100_overlap": overlap, "rmse": rmse})
        frame = pd.DataFrame(per_condition)
        records.append({
            "method": name,
            "mean_pearson": float(frame["pearson"].mean()),
            "mean_top100_overlap": float(frame["top100_overlap"].mean()),
            "mean_rmse": float(frame["rmse"].mean()),
            "n_test_conditions": int(len(frame)),
        })
        print(json.dumps(records[-1]))
        del matrix, train_response, train_feature, test_feature, model, predicted

    summary = pd.DataFrame(records)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output / "summary.parquet", index=False)
    report = {
        "design": "held-out perturbation response prediction; ridge on target-gene features (control mean, dropout, variance); reference = uncorrupted truth on test cells",
        "test_fraction": args.test_fraction,
        "alpha": args.alpha,
        "seed": args.seed,
        "train_conditions": len(train_conditions),
        "test_conditions": test_conditions,
        "results": summary.to_dict(orient="records"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
