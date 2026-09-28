#!/usr/bin/env python3
"""Attribute calibrated zero-selection performance to its feature families.

This is an attribution analysis, not a replacement for the deployable
calibration protocol in ``calibrated_selective_fill.py``.  Every learned
ranker is fit on the requested calibration split.  On the locked test split,
rankers are compared at exactly matched score coverage without consulting the
test labels.  All selected entries are filled with the same fused posterior
mean unless the explicit ``teacher_value`` value ablation is requested.

The output contains pooled ranking diagnostics, biological-unit metrics,
cluster-bootstrap summaries, and paired comparisons against the full
selector.  It deliberately calls non-masked test zeros "unlabelled" rather
than biological negatives: the artificial mask certifies positives only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


# The selector sees each teacher's proposal (log1p count scale) and three
# context features. Teacher columns are named "teacher:<name>".
CONTEXT_FEATURES = ("gene_mean", "gene_dropout", "library_size")

LEARNED_VARIANTS = ("full", "teacher_only", "context_only", "gene_mean_only", "gene_dropout_only")


CONDITION_FEATURES = ("condition_gene_mean", "condition_gene_dropout")


def teacher_feature_names(teacher_names, condition: bool = False) -> tuple[str, ...]:
    return tuple(f"teacher:{name}" for name in teacher_names) + CONTEXT_FEATURES + (CONDITION_FEATURES if condition else ())


def variant_columns(variant: str, feature_names) -> list[int]:
    names = list(feature_names)
    if variant == "full":
        return list(range(len(names)))
    if variant == "teacher_only":
        return [i for i, name in enumerate(names) if name.startswith("teacher:")]
    if variant == "context_only":
        return [i for i, name in enumerate(names) if not name.startswith("teacher:")]
    if variant == "gene_mean_only":
        return [names.index("gene_mean")]
    if variant == "gene_dropout_only":
        return [names.index("gene_dropout")]
    raise ValueError(f"unknown selector variant: {variant}")


def selector_features(teacher_values: np.ndarray, gene_mean: np.ndarray, gene_dropout: np.ndarray, library_size: np.ndarray,
                      condition_context: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """Stack log1p teacher proposals (one row per teacher) with the context features.

    ``condition_context`` optionally adds the gene's mean and zero fraction within
    the cell's condition (for example its perturbation).
    """
    columns = [np.log1p(np.maximum(teacher_values, 0.0)).T, gene_mean, gene_dropout, library_size]
    if condition_context is not None:
        columns += list(condition_context)
    return np.column_stack(columns).astype(np.float32)


DEFAULT_VARIANTS = tuple(LEARNED_VARIANTS) + ("random",)

ARCHITECTURES = (
    "logistic",
    "hist_gbdt",
    "extra_trees",
    "mlp",
    "ft_transformer",
)

ENSEMBLE_ARCHITECTURES = (
    "logistic",
    "hist_gbdt",
    "extra_trees",
    "mlp",
)


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def exact_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Return an exact-size deterministic top-k mask, including under ties."""
    scores = np.asarray(scores, dtype=float)
    k = max(0, min(int(k), len(scores)))
    selected = np.zeros(len(scores), dtype=bool)
    if k == 0:
        return selected
    if k == len(scores):
        selected[:] = True
        return selected
    cutoff = float(np.partition(scores, len(scores) - k)[len(scores) - k])
    selected[scores > cutoff] = True
    remaining = k - int(selected.sum())
    if remaining:
        ties = np.flatnonzero(scores == cutoff)
        selected[ties[:remaining]] = True
    return selected


def normalized_ranks(scores: np.ndarray) -> np.ndarray:
    """Map scores to average percentile ranks for scale-free ensembling."""
    if not len(scores):
        return np.asarray(scores, dtype=np.float32)
    return (rankdata(scores, method="average") / len(scores)).astype(np.float32)


@dataclass
class Candidates:
    rows: np.ndarray
    cols: np.ndarray
    labels: np.ndarray
    features: np.ndarray
    fused_values: np.ndarray
    teacher_values: np.ndarray  # one row per teacher contract
    truth_values: np.ndarray
    feature_names: tuple = ()


def candidate_frame(
    counts: np.ndarray,
    split_mask: np.ndarray,
    split_name: str,
    coordinates: pd.DataFrame,
    fused_mean: np.ndarray,
    teachers: list[np.ndarray],
    teacher_names: list[str],
    gene_mean: np.ndarray,
    gene_dropout: np.ndarray,
    library_size: np.ndarray,
    max_rows: int | None,
    seed: int,
) -> Candidates:
    rows, cols = np.where((counts == 0) & split_mask[:, None])
    # The coordinate table can carry labels from an earlier locked split.
    # Cross-fit folds must derive membership from their current split file.
    split_cells = np.flatnonzero(split_mask)
    split_coordinates = coordinates.loc[
        coordinates["cell_index"].astype(int).isin(split_cells)
    ]
    positive_rows = split_coordinates["cell_index"].to_numpy(dtype=np.int64)
    positive_cols = split_coordinates["gene_index"].to_numpy(dtype=np.int64)
    if "original_value" not in split_coordinates:
        raise ValueError("coordinates require original_value for attribution")
    positive_values = split_coordinates["original_value"].to_numpy(dtype=np.float32)
    if not len(positive_rows):
        raise ValueError(f"current split has no masked-positive coordinates: {split_name}")
    n_genes = counts.shape[1]
    positive_keys = positive_rows * n_genes + positive_cols
    order = np.argsort(positive_keys)
    positive_keys = positive_keys[order]
    positive_values = positive_values[order]
    if len(np.unique(positive_keys)) != len(positive_keys):
        raise ValueError(f"masked coordinates are not unique in {split_name}")
    if np.any(counts[positive_rows, positive_cols] != 0):
        raise ValueError(f"masked coordinates are not zero in corrupted counts for {split_name}")

    candidate_keys = rows.astype(np.int64) * n_genes + cols.astype(np.int64)

    if max_rows is not None and len(rows) > max_rows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(rows), size=max_rows, replace=False))
        candidate_keys = np.union1d(candidate_keys[keep], positive_keys)
        rows = candidate_keys // n_genes
        cols = candidate_keys % n_genes

    locations = np.searchsorted(positive_keys, candidate_keys)
    clipped = np.minimum(locations, max(len(positive_keys) - 1, 0))
    labels = (
        (locations < len(positive_keys))
        & (positive_keys[clipped] == candidate_keys)
    )
    truth_values = np.zeros(len(candidate_keys), dtype=np.float32)
    truth_values[labels] = positive_values[locations[labels]]
    labels = labels.astype(np.int8)
    teacher_stack = np.stack([matrix[rows, cols] for matrix in teachers])
    features = selector_features(teacher_stack, gene_mean[cols], gene_dropout[cols], library_size[rows])
    return Candidates(
        rows=rows,
        cols=cols,
        labels=labels,
        features=features,
        fused_values=fused_mean[rows, cols].astype(np.float32, copy=False),
        teacher_values=teacher_stack.astype(np.float32, copy=False),
        truth_values=truth_values,
        feature_names=teacher_feature_names(teacher_names),
    )


def predict_scores_in_batches(
    model,
    scaler: StandardScaler | None,
    features: np.ndarray,
    indices: np.ndarray,
    batch_rows: int,
) -> np.ndarray:
    scores = np.empty(len(features), dtype=np.float32)
    for start in range(0, len(features), batch_rows):
        stop = min(start + batch_rows, len(features))
        batch = features[start:stop][:, indices]
        if scaler is not None:
            batch = scaler.transform(batch)
        scores[start:stop] = model.predict_proba(batch)[:, 1]
    return scores


def stratified_fit_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    """Keep every positive when possible and sample negatives deterministically."""
    labels = np.asarray(labels)
    if len(labels) <= maximum:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if len(positive) >= maximum:
        return np.sort(rng.choice(positive, size=maximum, replace=False))
    keep_negative = rng.choice(
        negative, size=maximum - len(positive), replace=False,
    )
    return np.sort(np.concatenate([positive, keep_negative]))


def balanced_sample_weight(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if not positives or not negatives:
        raise ValueError("selector fitting requires both classes")
    weights = np.empty(len(labels), dtype=np.float64)
    weights[labels == 1] = len(labels) / (2.0 * positives)
    weights[labels == 0] = len(labels) / (2.0 * negatives)
    return weights


def fit_scores(
    variant: str,
    architecture: str,
    fit: Candidates,
    test: Candidates,
    max_fit_rows: int,
    score_batch_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if variant == "random":
        rng = np.random.default_rng(seed)
        return rng.random(len(fit.labels)), rng.random(len(test.labels)), {
            "kind": "random",
            "features": [],
        }
    if variant not in LEARNED_VARIANTS:
        raise ValueError(f"unknown selector variant: {variant}")
    indices = np.asarray(variant_columns(variant, fit.feature_names), dtype=int)
    names = [fit.feature_names[i] for i in indices]
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown selector architecture: {architecture}")
    fit_index = stratified_fit_indices(fit.labels, max_fit_rows, seed)
    train_features = fit.features[fit_index][:, indices]
    train_labels = fit.labels[fit_index]
    sample_weight = balanced_sample_weight(train_labels)
    scaler: StandardScaler | None = None
    fit_started = time.perf_counter()
    if architecture == "logistic":
        scaler = StandardScaler().fit(train_features)
        model = LogisticRegression(max_iter=2000, C=0.1, class_weight=None)
        model.fit(
            scaler.transform(train_features), train_labels,
            sample_weight=sample_weight,
        )
    elif architecture == "hist_gbdt":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
        )
        model.fit(train_features, train_labels, sample_weight=sample_weight)
    elif architecture == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=160,
            max_depth=18,
            min_samples_leaf=20,
            max_features=1.0,
            class_weight=None,
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train_features, train_labels, sample_weight=sample_weight)
    elif architecture == "mlp":
        scaler = StandardScaler().fit(train_features)
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=4096,
            learning_rate_init=1e-3,
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=seed,
        )
        model.fit(
            scaler.transform(train_features), train_labels,
            sample_weight=sample_weight,
        )
    else:
        try:
            from safefusion_benchmark.torch_selector import FTTransformerClassifier
        except ImportError as error:
            raise RuntimeError(
                "ft_transformer requires the optional PyTorch environment"
            ) from error

        scaler = StandardScaler().fit(train_features)
        model = FTTransformerClassifier(
            token_dim=32,
            heads=4,
            layers=2,
            dropout=0.1,
            batch_size=4096,
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_epochs=30,
            patience=6,
            validation_fraction=0.1,
            random_state=seed,
        )
        model.fit(
            scaler.transform(train_features), train_labels,
            sample_weight=sample_weight,
        )
    fit_seconds = time.perf_counter() - fit_started
    score_started = time.perf_counter()
    fit_scores_value = predict_scores_in_batches(
        model, scaler, fit.features, indices, score_batch_rows,
    )
    test_scores_value = predict_scores_in_batches(
        model, scaler, test.features, indices, score_batch_rows,
    )
    score_seconds = time.perf_counter() - score_started
    model_iterations = getattr(model, "n_iter_", None)
    if isinstance(model_iterations, np.ndarray):
        model_iterations = model_iterations.tolist()
    elif model_iterations is not None:
        model_iterations = int(model_iterations)
    return fit_scores_value, test_scores_value, {
        "kind": architecture,
        "features": list(names),
        "fit_rows": int(len(fit_index)),
        "fit_seconds": float(fit_seconds),
        "score_seconds": float(score_seconds),
        "model_iterations": model_iterations,
        "model_parameters": model.get_params(deep=False),
        **(
            {
                "coefficients": {
                    name: float(value) for name, value in zip(names, model.coef_[0])
                },
                "intercept": float(model.intercept_[0]),
            }
            if architecture == "logistic"
            else {}
        ),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    positives = float(frame["n_masked_positive"].sum())
    zeros = float(frame["n_zero_candidates"].sum())
    selected = float(frame["n_selected"].sum())
    selected_positive = float(frame["n_selected_masked_positive"].sum())
    raw_error = float(frame["raw_abs_error_sum"].sum())
    method_error = float(frame["method_abs_error_sum"].sum())
    raw_mae = safe_ratio(raw_error, positives)
    method_mae = safe_ratio(method_error, positives)
    return {
        "masked_log1p_mae": method_mae,
        "masked_error_recovery_fraction": safe_ratio(raw_mae - method_mae, raw_mae),
        "masked_recall": safe_ratio(selected_positive, positives),
        "selected_masked_precision": safe_ratio(selected_positive, selected),
        "fill_fraction": safe_ratio(selected, zeros),
    }


def summarize_with_bootstrap(
    units: pd.DataFrame,
    bootstrap: int,
    seed: int,
    reference_variant: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    comparisons: list[dict] = []
    unit_order = sorted(units["unit"].unique())
    grouped = {}
    for key, value in units.groupby(["variant", "value_source", "budget"], sort=True):
        frame = value.set_index("unit").reindex(unit_order).reset_index()
        if frame["n_zero_candidates"].isna().any():
            raise ValueError(f"selector attribution is missing biological units for {key}")
        grouped[key] = frame
    # Every variant is evaluated on the same locked units.  Reusing the exact
    # bootstrap indices is required for valid paired differences.
    shared_indices = rng.integers(0, len(unit_order), size=(bootstrap, len(unit_order)))
    bootstrap_cache: dict[tuple[str, str, float], dict[str, np.ndarray]] = {}
    for key, frame in grouped.items():
        variant, value_source, budget = key
        observed = aggregate_metrics(frame)
        samples = {metric: [] for metric in observed}
        for index in shared_indices:
            metrics = aggregate_metrics(frame.iloc[index])
            for metric, value in metrics.items():
                samples[metric].append(value)
        sample_arrays = {metric: np.asarray(values) for metric, values in samples.items()}
        bootstrap_cache[key] = sample_arrays
        for metric, value in observed.items():
            records.append({
                "variant": variant,
                "value_source": value_source,
                "budget": float(budget),
                "metric": metric,
                "estimate": float(value),
                "ci_low": float(np.nanquantile(sample_arrays[metric], 0.025)),
                "ci_high": float(np.nanquantile(sample_arrays[metric], 0.975)),
                "n_units": int(len(frame)),
            })

    for key, samples in bootstrap_cache.items():
        variant, value_source, budget = key
        reference_key = (reference_variant, "fusion_value", budget)
        if key == reference_key or reference_key not in bootstrap_cache:
            continue
        reference_samples = bootstrap_cache[reference_key]
        observed = aggregate_metrics(grouped[key])
        reference_observed = aggregate_metrics(grouped[reference_key])
        for metric in observed:
            # Positive means the candidate variant is larger than full.
            difference = samples[metric] - reference_samples[metric]
            comparisons.append({
                "variant": variant,
                "value_source": value_source,
                "reference": f"{reference_variant}__fusion_value",
                "budget": float(budget),
                "metric": metric,
                "difference": float(observed[metric] - reference_observed[metric]),
                "ci_low": float(np.nanquantile(difference, 0.025)),
                "ci_high": float(np.nanquantile(difference, 0.975)),
                "n_units": int(len(grouped[key])),
            })
    return pd.DataFrame(records), pd.DataFrame(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--fusion-contract", required=True)
    parser.add_argument("--teacher-contract", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-split", default="validation", choices=["development", "validation"])
    parser.add_argument("--unit-column", required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.02])
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument(
        "--architectures", nargs="+", choices=ARCHITECTURES,
        default=["logistic"],
        help=(
            "Selector model families. Multiple architectures are normally "
            "combined with --variants full for an architecture benchmark."
        ),
    )
    parser.add_argument(
        "--rank-ensemble", action="store_true",
        help=(
            "Also evaluate the mean percentile rank across the requested "
            "architectures. Requires at least two learned full selectors."
        ),
    )
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth_adata = ad.read_h5ad(args.truth, backed="r")
    if args.unit_column not in truth_adata.obs:
        raise ValueError(f"unit column is absent from truth data: {args.unit_column}")
    counts = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    if not np.array_equal(
        corrupted_adata.obs_names.astype(str), truth_adata.obs_names.astype(str)
    ):
        raise ValueError("truth and corrupted cell order differ")
    splits = pd.read_parquet(args.splits).set_index("cell_id").loc[
        corrupted_adata.obs_names.astype(str), "split"
    ].to_numpy()
    fit_mask = splits == args.fit_split
    test_mask = splits == "test"
    coordinates = pd.read_parquet(args.coordinates)

    fusion_path = Path(args.fusion_contract)
    fused_mean = np.load(fusion_path / "mean.npy", allow_pickle=False, mmap_mode="r")
    teachers = [
        np.load(Path(path) / "mean.npy", allow_pickle=False, mmap_mode="r")
        for path in args.teacher_contract
    ]
    teacher_names = [Path(path).name for path in args.teacher_contract]
    gene_mean = np.log1p(np.mean(counts[fit_mask], axis=0))
    gene_dropout = np.mean(counts[fit_mask] <= 0, axis=0)
    library_size = np.log1p(counts.sum(axis=1))

    fit = candidate_frame(
        counts, fit_mask, args.fit_split, coordinates, fused_mean,
        teachers, teacher_names, gene_mean, gene_dropout, library_size,
        args.max_fit_rows, args.seed,
    )
    test = candidate_frame(
        counts, test_mask, "test", coordinates, fused_mean,
        teachers, teacher_names, gene_mean, gene_dropout, library_size,
        None, args.seed + 1,
    )
    if not fit.labels.any() or not test.labels.any():
        raise ValueError("both calibration and test candidates require masked positives")

    unit_values = truth_adata.obs[args.unit_column].astype(str).to_numpy()[test.rows]
    ranking_records: list[dict] = []
    unit_records: list[dict] = []
    model_reports: dict[str, dict] = {}
    score_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    jobs: list[tuple[str, str, str]] = []
    for variant in args.variants:
        if variant == "random":
            jobs.append((variant, "random", variant))
            continue
        for architecture in args.architectures:
            selector_id = (
                variant
                if args.architectures == ["logistic"]
                else f"{variant}__{architecture}"
            )
            jobs.append((variant, architecture, selector_id))

    reference_variant = (
        "full" if args.architectures == ["logistic"] else f"full__{args.architectures[0]}"
    )

    def record_selector(
        selector_id: str,
        feature_variant: str,
        architecture: str,
        fit_score: np.ndarray,
        test_score: np.ndarray,
    ) -> None:
        ranking_records.append({
            "variant": selector_id,
            "feature_variant": feature_variant,
            "architecture": architecture,
            "fit_roc_auc": float(roc_auc_score(fit.labels, fit_score)),
            "fit_pr_auc": float(average_precision_score(fit.labels, fit_score)),
            "test_roc_auc": float(roc_auc_score(test.labels, test_score)),
            "test_pr_auc": float(average_precision_score(test.labels, test_score)),
            "n_fit_candidates": int(len(fit.labels)),
            "n_test_candidates": int(len(test.labels)),
            "n_fit_masked_positive": int(fit.labels.sum()),
            "n_test_masked_positive": int(test.labels.sum()),
        })
        for budget in args.budgets:
            selected = exact_topk(test_score, round(float(budget) * len(test_score)))
            # The reference selector also inserts each teacher's own value into
            # the same selected zeros, which isolates the value from the ranking.
            value_sources = {"fusion_value": test.fused_values}
            if selector_id == reference_variant:
                for name, values in zip(teacher_names, test.teacher_values):
                    value_sources[f"teacher_{name}"] = values
            for value_source, candidate_values in value_sources.items():
                predicted = np.where(selected, np.maximum(candidate_values, 0.0), 0.0)
                for unit in np.unique(unit_values):
                    in_unit = unit_values == unit
                    positives = in_unit & (test.labels == 1)
                    selected_unit = in_unit & selected
                    truth_values = test.truth_values[positives]
                    predicted_values = predicted[positives]
                    unit_records.append({
                        "unit": unit,
                        "variant": selector_id,
                        "value_source": value_source,
                        "budget": float(budget),
                        "n_zero_candidates": int(in_unit.sum()),
                        "n_masked_positive": int(positives.sum()),
                        "n_selected": int(selected_unit.sum()),
                        "n_selected_masked_positive": int(np.sum(selected_unit & (test.labels == 1))),
                        "raw_abs_error_sum": float(np.log1p(truth_values).sum()),
                        "method_abs_error_sum": float(np.abs(
                            np.log1p(truth_values) - np.log1p(predicted_values)
                        ).sum()),
                    })

    for job_index, (feature_variant, architecture, selector_id) in enumerate(jobs):
        fit_score, test_score, model_report = fit_scores(
            feature_variant, architecture, fit, test,
            args.max_fit_rows, args.score_batch_rows,
            args.seed + 100 * job_index,
        )
        model_reports[selector_id] = model_report
        record_selector(
            selector_id, feature_variant, architecture, fit_score, test_score,
        )
        if args.rank_ensemble and feature_variant == "full" and architecture != "random":
            score_cache[selector_id] = (fit_score, test_score)

    if args.rank_ensemble:
        if len(score_cache) < 2:
            raise ValueError("--rank-ensemble requires at least two learned full selectors")
        ensemble_id = "full__rank_ensemble"
        ensemble_started = time.perf_counter()
        ensemble_fit_score = np.mean(
            [normalized_ranks(values[0]) for values in score_cache.values()], axis=0,
        ).astype(np.float32)
        ensemble_test_score = np.mean(
            [normalized_ranks(values[1]) for values in score_cache.values()], axis=0,
        ).astype(np.float32)
        ensemble_seconds = time.perf_counter() - ensemble_started
        model_reports[ensemble_id] = {
            "kind": "rank_ensemble",
            "features": list(fit.feature_names),
            "members": list(score_cache),
            "score_seconds": float(ensemble_seconds),
        }
        record_selector(
            ensemble_id, "full", "rank_ensemble",
            ensemble_fit_score, ensemble_test_score,
        )

    unit_frame = pd.DataFrame(unit_records)
    summary, comparisons = summarize_with_bootstrap(
        unit_frame, args.bootstrap, args.seed + 10_000, reference_variant,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ranking_records).to_parquet(output / "ranking_metrics.parquet", index=False)
    unit_frame.to_parquet(output / "unit_metrics.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    report = {
        "design": "selector feature attribution at exact matched test-score coverage; no test labels used for selection",
        "fit_split": args.fit_split,
        "test_split": "test",
        "unit_column": args.unit_column,
        "budgets": [float(value) for value in args.budgets],
        "variants": list(args.variants),
        "architectures": list(args.architectures),
        "rank_ensemble": bool(args.rank_ensemble),
        "reference_selector": reference_variant,
        "feature_names": list(fit.feature_names),
        "value_ablation": "all selectors use fused values; full also evaluated with max-teacher values",
        "non_masked_zero_label": "unlabelled, not certified biological zero",
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        "models": model_reports,
        "artifacts": [
            "ranking_metrics.parquet", "unit_metrics.parquet",
            "bootstrap_summary.parquet", "paired_comparisons.parquet",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    truth_adata.file.close()


if __name__ == "__main__":
    main()
