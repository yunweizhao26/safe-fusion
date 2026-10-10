from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contracts import order_hash, write_output_contract
from .hashing import sha256_file


def _gene_statistic(counts: np.ndarray, statistic: str, training_mask: np.ndarray | None = None) -> np.ndarray:
    fit_counts = counts if training_mask is None else counts[training_mask]
    observed = fit_counts > 0
    result = np.zeros(counts.shape[1], dtype=np.float32)
    for gene in range(counts.shape[1]):
        values = fit_counts[observed[:, gene], gene]
        if len(values):
            result[gene] = np.mean(values) if statistic == "mean" else np.median(values)
    output = counts.astype(np.float32).copy()
    rows, cols = np.where(counts == 0)
    output[rows, cols] = result[cols]
    return output


def run_builtin(method: str, counts: np.ndarray, seed: int, training_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    rng = np.random.default_rng(seed)
    if method == "raw":
        return counts.astype(np.float32), None, np.zeros_like(counts, dtype=np.float32)
    if method in {"gene_mean", "gene_median"}:
        output = _gene_statistic(counts, method.removeprefix("gene_"), training_mask)
        residual = output - counts
        variance = np.broadcast_to(np.var(residual, axis=0, dtype=np.float64), counts.shape).astype(np.float32).copy()
        return output, variance, output.copy()
    mean_teacher = _gene_statistic(counts, "mean", training_mask)
    median_teacher = _gene_statistic(counts, "median", training_mask)
    teacher_stack = np.stack([mean_teacher, median_teacher])
    disagreement = np.var(np.log1p(teacher_stack), axis=0).astype(np.float32)
    fusion = np.mean(teacher_stack, axis=0, dtype=np.float32)
    if method == "random_fill":
        zero = counts == 0
        output = counts.astype(np.float32).copy()
        candidates = fusion[zero]
        output[zero] = candidates[rng.permutation(len(candidates))]
        return output, None, rng.random(counts.shape, dtype=np.float32)
    if method == "safe_fusion_no_gate":
        return fusion, disagreement, fusion
    confidence = -np.log(disagreement + 1e-6)
    if method == "safe_fusion_random_gate":
        confidence = rng.random(counts.shape, dtype=np.float32)
    elif method == "safe_fusion_variance_shuffled":
        confidence = confidence.ravel()[rng.permutation(confidence.size)].reshape(confidence.shape)
    cutoff = np.quantile(confidence[counts == 0], 0.90) if np.any(counts == 0) else np.inf
    gate = (counts == 0) & (confidence >= cutoff)
    output = counts.astype(np.float32).copy()
    if method != "safe_fusion_gate_no_fallback":
        output[counts == 0] = mean_teacher[counts == 0]
    output[gate] = fusion[gate]
    return output, disagreement, confidence.astype(np.float32)


def write_builtin_contract(
    method: str,
    counts: np.ndarray,
    cell_ids: list[str],
    gene_ids: list[str],
    output_directory: str | Path,
    input_path: str | Path,
    coordinates_path: str | Path,
    seed: int,
    training_mask: np.ndarray | None = None,
) -> None:
    mean, variance, fill_score = run_builtin(method, counts, seed, training_mask)
    metadata = {
        "method": method,
        "method_version": "safefusion-benchmark/0.1.0",
        "scale": "counts",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["development", "validation"],
        "parameters": {"gate_quantile": 0.90} if method.startswith("safe_fusion") else {},
        "seed": seed,
        "input_sha256": sha256_file(input_path),
        "coordinates_sha256": sha256_file(coordinates_path),
    }
    write_output_contract(output_directory, mean, metadata, variance=variance, fill_score=fill_score)
