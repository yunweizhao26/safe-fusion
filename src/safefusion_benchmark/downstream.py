from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .metrics import average_precision_tie_aware, log1p_mae, spearman


def _contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    _, true_index = np.unique(labels_true, return_inverse=True)
    _, pred_index = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((true_index.max() + 1, pred_index.max() + 1), dtype=np.int64)
    np.add.at(table, (true_index, pred_index), 1)
    return table


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(labels_true, labels_pred)
    comb = lambda x: x * (x - 1) / 2
    n = table.sum()
    if n < 2:
        return float("nan")
    sum_cells = comb(table).sum()
    sum_rows, sum_cols = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    expected = sum_rows * sum_cols / comb(n)
    maximum = 0.5 * (sum_rows + sum_cols)
    return float((sum_cells - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalized_mutual_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    table = _contingency(labels_true, labels_pred).astype(float)
    probabilities = table / table.sum()
    row, col = probabilities.sum(axis=1), probabilities.sum(axis=0)
    nonzero = probabilities > 0
    expected = row[:, None] * col[None, :]
    mutual = float(np.sum(probabilities[nonzero] * np.log(probabilities[nonzero] / expected[nonzero])))
    h_row = -float(np.sum(row[row > 0] * np.log(row[row > 0])))
    h_col = -float(np.sum(col[col > 0] * np.log(col[col > 0])))
    return mutual / np.sqrt(h_row * h_col) if h_row and h_col else 1.0


def kmeans(matrix: np.ndarray, n_clusters: int, seed: int, iterations: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(len(matrix), size=n_clusters, replace=False)].copy()
    labels = np.zeros(len(matrix), dtype=int)
    for _ in range(iterations):
        updated = np.argmin(cdist(matrix, centers), axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for cluster in range(n_clusters):
            if np.any(labels == cluster):
                centers[cluster] = matrix[labels == cluster].mean(axis=0)
    return labels


def neighborhood_purity(matrix: np.ndarray, labels: np.ndarray, neighbors: int = 10) -> float:
    k = min(neighbors, len(matrix) - 1)
    matches = 0
    total = 0
    for start in range(0, len(matrix), 256):
        stop = min(start + 256, len(matrix))
        distances = cdist(matrix[start:stop], matrix)
        distances[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        matches += int(np.sum(labels[nearest] == labels[start:stop, None]))
        total += int((stop - start) * k)
    return float(matches / total)


def randomized_pca_embedding(
    matrix: np.ndarray,
    train: np.ndarray,
    seed: int,
    components: int = 50,
) -> np.ndarray:
    """Library-normalize and fit a deterministic randomized PCA on train cells."""
    library = matrix.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    normalized = np.log1p(np.clip(matrix, 0, None) * scale[:, None]).astype(np.float32)
    center = normalized[train].mean(axis=0, dtype=np.float64).astype(np.float32)
    fit = normalized[train] - center
    k = min(components, fit.shape[0] - 1, fit.shape[1] - 1)
    oversampled = min(k + 10, fit.shape[0], fit.shape[1])
    omega = np.random.default_rng(seed).normal(size=(fit.shape[1], oversampled)).astype(np.float32)
    sketch = fit @ omega
    for _ in range(2):
        sketch = fit @ (fit.T @ sketch)
    q, _ = np.linalg.qr(sketch, mode="reduced")
    _, _, vt = np.linalg.svd(q.T @ fit, full_matrices=False)
    return ((normalized - center) @ vt[:k].T).astype(np.float32)


def macro_f1_nearest_centroid(matrix: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> float:
    classes = np.unique(labels[train])
    centroids = np.stack([matrix[train & (labels == cls)].mean(axis=0) for cls in classes])
    predicted = classes[np.argmin(cdist(matrix[test], centroids), axis=1)]
    actual = labels[test]
    f1 = []
    for cls in classes:
        tp = np.sum((predicted == cls) & (actual == cls))
        fp = np.sum((predicted == cls) & (actual != cls))
        fn = np.sum((predicted != cls) & (actual == cls))
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(f1))


def marker_recovery(matrix: np.ndarray, reference: np.ndarray, labels: np.ndarray, train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    aps, correlations = [], []
    for cls in np.unique(labels):
        in_train = train & (labels == cls)
        out_train = train & (labels != cls)
        if not in_train.any() or not out_train.any():
            continue
        ref_score = np.log1p(reference[in_train].mean(axis=0)) - np.log1p(reference[out_train].mean(axis=0))
        marker = ref_score >= np.quantile(ref_score, 0.9)
        in_test = test & (labels == cls)
        out_test = test & (labels != cls)
        pred_score = np.log1p(matrix[in_test].mean(axis=0)) - np.log1p(matrix[out_test].mean(axis=0))
        aps.append(average_precision_tie_aware(marker, pred_score))
        correlations.append(spearman(ref_score, pred_score))
    return float(np.mean(aps)), float(np.mean(correlations))


def pseudobulk_logfc(matrix: np.ndarray, obs: pd.DataFrame, unit: str, condition: str) -> np.ndarray:
    groups = []
    for _, block in obs.groupby(unit, sort=True, observed=True):
        levels = sorted(block[condition].astype(str).unique())
        if len(levels) != 2:
            continue
        first = matrix[obs.index.get_indexer(block.index[block[condition].astype(str) == levels[0]])].sum(axis=0)
        second = matrix[obs.index.get_indexer(block.index[block[condition].astype(str) == levels[1]])].sum(axis=0)
        groups.append(np.log2((second + 1) / (first + 1)))
    if not groups:
        raise ValueError("no biological unit contains both conditions")
    return np.mean(groups, axis=0)


def infer_grn_scores(matrix: np.ndarray, gene_ids: list[str], edges: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    expression = np.log1p(matrix).astype(float)
    centered = expression - expression.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered**2, axis=0))
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(centered.T @ centered, denominator, out=np.zeros((matrix.shape[1], matrix.shape[1])), where=denominator > 0)
    correlation = np.abs(correlation)
    index = {gene: i for i, gene in enumerate(gene_ids)}
    truth_edges = {tuple(edge) for edge in edges}
    labels, scores = [], []
    for source in gene_ids:
        for target in gene_ids:
            if source == target:
                continue
            labels.append(int((source, target) in truth_edges))
            scores.append(correlation[index[source], index[target]])
    return np.asarray(labels), np.asarray(scores)
