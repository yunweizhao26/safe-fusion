#!/usr/bin/env python3
"""Run leakage-safe classical teachers or the frozen Safe Fusion model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


REPOSITORY = Path(__file__).resolve().parents[1]
PARENT_REPOSITORY = REPOSITORY.parent
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(PARENT_REPOSITORY))

from safefusion_benchmark.contracts import order_hash, write_output_contract  # noqa: E402
from safefusion_benchmark.hashing import sha256_file  # noqa: E402


def log1p_cpm(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    library = counts.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(counts * scale[:, None]).astype(np.float32), library.astype(np.float32)


def gene_median(counts: np.ndarray, training: np.ndarray) -> np.ndarray:
    result = np.zeros(counts.shape[1], dtype=np.float32)
    fit = counts[training]
    for gene in range(counts.shape[1]):
        values = fit[fit[:, gene] > 0, gene]
        if len(values):
            result[gene] = np.median(values)
    output = counts.copy()
    rows, cols = np.where(output == 0)
    output[rows, cols] = result[cols]
    return output.astype(np.float32)


def svd_teacher(counts: np.ndarray, training: np.ndarray, seed: int, components: int) -> tuple[np.ndarray, PCA]:
    normalized, library = log1p_cpm(counts)
    n_components = min(components, training.sum() - 1, counts.shape[1] - 1)
    model = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    model.fit(normalized[training])
    reconstructed = model.inverse_transform(model.transform(normalized))
    reconstructed = np.expm1(np.clip(reconstructed, 0.0, 20.0))
    output = reconstructed * (library[:, None] / 1e4)
    return np.clip(output, 0.0, None).astype(np.float32), model


def graph_teacher(
    counts: np.ndarray,
    training: np.ndarray,
    pca: PCA,
    neighbors: int,
) -> np.ndarray:
    normalized, _ = log1p_cpm(counts)
    embedding = pca.transform(normalized)
    fit_embedding = embedding[training]
    fit_counts = counts[training]
    k = min(neighbors, len(fit_embedding))
    model = NearestNeighbors(n_neighbors=k, metric="euclidean")
    model.fit(fit_embedding)
    distances, indices = model.kneighbors(embedding)
    # Smooth distance weights, with exact matches retaining finite dominance.
    positive = distances[distances > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    weights = np.exp(-np.square(distances / max(bandwidth, 1e-6)))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    output = np.empty_like(counts, dtype=np.float32)
    for start in range(0, len(counts), 128):
        stop = min(start + 128, len(counts))
        output[start:stop] = np.einsum(
            "nk,nkg->ng",
            weights[start:stop],
            fit_counts[indices[start:stop]],
            optimize=True,
        )
    return output.astype(np.float32)


def fusion_prediction(
    counts: np.ndarray,
    training: np.ndarray,
    teachers: dict[str, np.ndarray],
    normalized: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    import torch

    from fusion.eval import predict_latent_truth
    from fusion.train import TrainConfig, train_latent_truth

    torch.manual_seed(seed)
    np.random.seed(seed)
    fit_counts = counts[training].astype(np.float32)
    fit_teachers = {name: value[training].astype(np.float32) for name, value in teachers.items()}
    gene_mean = np.log1p(np.mean(fit_counts, axis=0)).astype(np.float32)
    gene_dropout = np.mean(fit_counts <= 0, axis=0).astype(np.float32)

    pca_components = min(30, fit_counts.shape[0] - 1, fit_counts.shape[1] - 1)
    feature_pca = PCA(n_components=pca_components, svd_solver="randomized", random_state=seed)
    fit_pca = feature_pca.fit_transform(normalized[training]).astype(np.float32)
    all_pca = feature_pca.transform(normalized).astype(np.float32)
    config = TrainConfig(
        batch_size=batch_size,
        epochs=epochs,
        lr=1e-3,
        mask_fraction=0.15,
        teacher_weight=1.0,
        best_teacher_weight=0.8,
        best_teacher_min_log=0.0,
        best_teacher_temp=0.5,
        teacher_warmup_epochs=1,
        teacher_ramp_epochs=3,
        teacher_dropout=0.4,
        teacher_loss_on_prior=True,
        teacher_calibration_weight=1e-3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model, history = train_latent_truth(
        fit_counts,
        fit_teachers,
        config,
        gene_mean=gene_mean,
        gene_dropout=gene_dropout,
        pca_features=fit_pca,
        pca_proj_dim=8,
        seed=seed,
    )
    prediction, dropout, variance_log = predict_latent_truth(
        model,
        counts.astype(np.float32),
        teachers={name: value.astype(np.float32) for name, value in teachers.items()},
        fuse=True,
        cell_loglib=np.log1p(counts.sum(axis=1)).astype(np.float32),
        gene_mean=gene_mean,
        gene_dropout=gene_dropout,
        pca_features=all_pca,
        batch_size=batch_size,
        device=config.device,
    )
    # Delta-method conversion from posterior log1p variance to count variance.
    variance_count = variance_log * np.square(1.0 + prediction)
    details = {
        "train_config": config.__dict__,
        "history": history,
        "device": config.device,
        "teacher_names": list(teachers),
        "pca_components": pca_components,
    }
    return prediction.astype(np.float32), variance_count.astype(np.float32), dropout.astype(np.float32), details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["gene_median", "svd_impute", "graph_smooth", "safe_fusion"],
        required=True,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    split_frame = pd.read_parquet(args.splits).set_index("cell_id")
    split = split_frame.loc[adata.obs_names.astype(str), "split"].to_numpy()
    training = np.isin(split, ["development", "validation"])
    if not training.any() or np.any(split[training] == "test"):
        raise ValueError("invalid training split")

    normalized, _ = log1p_cpm(counts)
    median = gene_median(counts, training)
    svd = None
    pca = None
    graph = None
    if args.method != "gene_median":
        svd, pca = svd_teacher(counts, training, args.seed, args.components)
    if args.method in {"graph_smooth", "safe_fusion"}:
        assert pca is not None
        graph = graph_teacher(counts, training, pca, args.neighbors)
    details: dict = {
        "fit_cells": int(training.sum()),
        "heldout_test_cells": int((split == "test").sum()),
        "svd_components": int(pca.n_components_) if pca is not None else None,
        "graph_neighbors": int(args.neighbors) if graph is not None else None,
        "test_used_for_fit": False,
    }
    variance = None
    fill_score = None
    if args.method == "gene_median":
        prediction = median
    elif args.method == "svd_impute":
        assert svd is not None
        prediction = svd
    elif args.method == "graph_smooth":
        assert graph is not None
        prediction = graph
    else:
        assert svd is not None and graph is not None
        prediction, variance, dropout, fusion_details = fusion_prediction(
            counts,
            training,
            {"gene_median": median, "svd_impute": svd, "graph_smooth": graph},
            normalized,
            args.seed,
            args.epochs,
            args.batch_size,
        )
        fill_score = -variance
        details.update(fusion_details)
        details["dropout_probability_summary"] = {
            "mean": float(dropout.mean()),
            "minimum": float(dropout.min()),
            "maximum": float(dropout.max()),
        }

    try:
        version = subprocess.check_output(
            ["git", "-C", str(PARENT_REPOSITORY), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        version = "unavailable"
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    metadata = {
        "method": args.method,
        "method_version": version,
        "scale": "counts",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["development", "validation"],
        "training_data": {"cell_ids_sha256": order_hash(np.asarray(cell_ids)[training].tolist())},
        "parameters": details,
        "seed": args.seed,
        "input_sha256": sha256_file(args.input),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, prediction, metadata, variance=variance, fill_score=fill_score)
    print(json.dumps({"method": args.method, "shape": list(prediction.shape), "parameters": details}, default=str))


if __name__ == "__main__":
    main()
