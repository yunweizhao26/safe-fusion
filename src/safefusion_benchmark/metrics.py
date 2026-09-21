from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import ndtr


def log1p_mae(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.log1p(truth) - np.log1p(np.clip(prediction, 0, None)))))


def poisson_deviance(truth: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(truth, dtype=float)
    mu = np.clip(np.asarray(prediction, dtype=float), 1e-10, None)
    terms = mu.copy()
    positive = y > 0
    terms[positive] = y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive])
    return float(2.0 * np.mean(terms))


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(truth: np.ndarray, prediction: np.ndarray) -> float:
    if len(truth) < 2:
        return float("nan")
    x, y = rankdata_average(np.asarray(truth)), rankdata_average(np.asarray(prediction))
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def roc_auc_tie_aware(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def average_precision_tie_aware(labels: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-group AP; an all-tied score equals prevalence exactly."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    frame = pd.DataFrame({"label": labels, "score": scores})
    groups = frame.groupby("score", sort=True)["label"].agg(["sum", "count"]).sort_index(ascending=False)
    tp, seen, ap = 0.0, 0.0, 0.0
    for row in groups.itertuples():
        tp += float(row.sum)
        seen += float(row.count)
        recall_increment = float(row.sum) / positives
        ap += recall_increment * (tp / seen)
    return float(ap)


def expected_budget_metrics(labels: np.ndarray, scores: np.ndarray, budget: float) -> dict[str, float]:
    """Expected metrics when the cutoff intersects a tie group."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_select = min(len(labels), max(0, int(round(float(budget) * len(labels)))))
    if n_select == 0:
        return {"precision": float("nan"), "recall": 0.0, "f1": 0.0, "false_fill_rate": 0.0}
    order = np.argsort(-scores, kind="mergesort")
    cutoff = scores[order[n_select - 1]]
    above = scores > cutoff
    tied = scores == cutoff
    slots = n_select - int(above.sum())
    expected_tp = float(labels[above].sum())
    if tied.sum():
        expected_tp += slots * float(labels[tied].mean())
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    precision = expected_tp / n_select
    recall = expected_tp / positives if positives else float("nan")
    false_fill_rate = (n_select - expected_tp) / negatives if negatives else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "false_fill_rate": false_fill_rate}


def risk_coverage_auc(truth: np.ndarray, prediction: np.ndarray, confidence: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    confidence = np.asarray(confidence, dtype=float)
    order = np.argsort(-confidence, kind="mergesort")
    losses = np.abs(np.log1p(truth[order]) - np.log1p(np.clip(prediction[order], 0, None)))
    risk = np.cumsum(losses) / np.arange(1, len(losses) + 1)
    coverage = np.arange(1, len(losses) + 1) / len(losses)
    return float(np.trapz(risk, coverage))


def interval_coverage(truth: np.ndarray, mean: np.ndarray, variance: np.ndarray, z: float = 1.96) -> float:
    sd = np.sqrt(np.clip(variance, 0, None))
    return float(np.mean((truth >= mean - z * sd) & (truth <= mean + z * sd)))


def gaussian_nll(truth: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    variance = np.clip(np.asarray(variance, dtype=float), 1e-8, None)
    residual = np.asarray(truth, dtype=float) - np.asarray(mean, dtype=float)
    return float(np.mean(0.5 * (np.log(2 * np.pi * variance) + residual**2 / variance)))


def gaussian_crps(truth: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    sd = np.sqrt(np.clip(np.asarray(variance, dtype=float), 1e-8, None))
    z = (np.asarray(truth, dtype=float) - np.asarray(mean, dtype=float)) / sd
    phi = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    value = sd * (z * (2 * ndtr(z) - 1) + 2 * phi - 1 / np.sqrt(np.pi))
    return float(np.mean(value))


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    method: str,
    baseline: str,
    value_column: str,
    unit_column: str,
    replicates: int,
    seed: int,
    weight_column: str | None = None,
) -> dict[str, float]:
    selected = frame[frame["method"].isin([method, baseline])]
    pivot = selected.pivot_table(index=unit_column, columns="method", values=value_column, aggfunc="mean").dropna()
    if method not in pivot or baseline not in pivot or pivot.empty:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_units": 0}
    differences = (pivot[method] - pivot[baseline]).to_numpy()
    weights = None
    if weight_column and weight_column in frame:
        weight_pivot = selected.pivot_table(index=unit_column, columns="method", values=weight_column, aggfunc="first").dropna()
        if method in weight_pivot and baseline in weight_pivot and len(weight_pivot) == len(pivot):
            weights = weight_pivot[method].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled = rng.choice(len(differences), size=len(differences), replace=True)
        if weights is not None:
            samples[index] = np.average(differences[sampled], weights=weights[sampled])
        else:
            samples[index] = differences[sampled].mean()
    return {
        "difference": float(np.average(differences, weights=weights) if weights is not None else differences.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n_units": int(len(differences)),
    }


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    n = len(values)
    for rank, index in enumerate(order):
        running = max(running, (n - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted
