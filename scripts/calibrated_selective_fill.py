#!/usr/bin/env python3
"""Validation-calibrated selective zero filling for Safe Fusion.

Ranks zero entries by a selector trained to recognise masked positives in the
selector-fitting cells, using the teacher proposals and gene and cell context.
Writes one output contract per fill budget: raw counts plus the top-budget
zeros replaced by the Safe Fusion stacked value.
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
from sklearn.metrics import roc_auc_score, average_precision_score


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.contracts import write_output_contract  # noqa: E402
from selector_attribution import (  # noqa: E402
    ARCHITECTURES,
    Candidates,
    ENSEMBLE_ARCHITECTURES,
    exact_topk,
    fit_scores,
    selector_features,
    teacher_feature_names,
)


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def score_percentile_from_fit(
    fit_scores_value: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Map values through the empirical fit-score CDF without test fitting."""
    ordered = np.sort(np.asarray(fit_scores_value, dtype=np.float32))
    return (
        np.searchsorted(ordered, values, side="right") / max(1, len(ordered))
    ).astype(np.float32)


def detection_probability(p: np.ndarray, mask_rate: float) -> np.ndarray:
    """Probability that a candidate zero was a detected entry before masking.

    Each detected entry is hidden with probability ``mask_rate``, and a
    candidate zero is either a hidden detection or an undetected entry, so
    p = rho*d / (rho*d + 1 - d) and d = p / (p + rho*(1 - p)).
    """
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return p / np.maximum(p + mask_rate * (1.0 - p), 1e-12)


def cross_fitted_calibration(
    architecture: str,
    fit_rows: np.ndarray,
    fit_labels: np.ndarray,
    fit_features: np.ndarray,
    negative_weight: float,
    folds: int,
    max_fit_rows: int,
    score_batch_rows: int,
    seed: int,
    make_candidates,
):
    """Isotonic map from selector score to P(masked positive).

    The selector is refitted ``folds`` times, each time without one group of
    fitting cells, and scores the held-out cells. The isotonic map is fitted
    on these out-of-fold scores. Negatives are weighted by ``negative_weight``
    to undo any subsampling of fitting candidates. No test entry is used.
    """
    from sklearn.isotonic import IsotonicRegression

    cells = np.unique(fit_rows)
    fold_of_cell = np.random.default_rng(seed).permutation(len(cells)) % folds
    fold = fold_of_cell[np.searchsorted(cells, fit_rows)]
    out_of_fold = np.empty(len(fit_labels), dtype=np.float32)
    for index in range(folds):
        held = fold == index
        _, held_score, _ = fit_scores(
            "full", architecture,
            make_candidates(fit_labels[~held], fit_features[~held]),
            make_candidates(fit_labels[held], fit_features[held]),
            max_fit_rows, score_batch_rows, seed + 1000 + index,
        )
        out_of_fold[held] = held_score
    weight = np.where(fit_labels == 1, 1.0, negative_weight)
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(out_of_fold, fit_labels.astype(np.float64), sample_weight=weight)
    return isotonic, out_of_fold, weight


def reliability_table(probability: np.ndarray, labels: np.ndarray, weight: np.ndarray | None = None, bins: int = 10) -> list[dict]:
    weight = np.ones(len(labels)) if weight is None else weight
    edges = np.quantile(probability, np.linspace(0.0, 1.0, bins + 1))
    index = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, bins - 1)
    table = []
    for b in range(bins):
        keep = index == b
        if not keep.any():
            continue
        w = weight[keep]
        table.append({
            "bin": b,
            "n": int(keep.sum()),
            "mean_predicted": float(np.average(probability[keep], weights=w)),
            "observed_rate": float(np.average(labels[keep], weights=w)),
        })
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--fusion-contract", required=True)
    parser.add_argument("--teacher-contract", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.02, 0.05, 0.081, 0.10])
    parser.add_argument("--curve-min-budget", type=float, default=0.001)
    parser.add_argument("--curve-max-budget", type=float, default=1.0)
    parser.add_argument("--curve-points", type=int, default=1000)
    parser.add_argument("--fit-split", default="validation", choices=["development", "validation"])
    parser.add_argument("--apply-split", default="test", choices=["test"])
    parser.add_argument("--fit-cells", type=int, default=None)
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument(
        "--score-gene",
        nargs="+",
        default=None,
        help=(
            "Optionally write final selector scores for zero entries of these "
            "genes in the fit and test splits. This does not change training."
        ),
    )
    parser.add_argument(
        "--architecture",
        choices=[*ARCHITECTURES, "rank_ensemble"],
        default="logistic",
        help="Calibrated zero-selector model; logistic preserves the original default.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=["calibration_threshold", "apply_topk"],
        default="calibration_threshold",
        help=(
            "Transfer the fit threshold (original default), or apply an exact "
            "target-score budget without using target labels."
        ),
    )
    parser.add_argument(
        "--detection-rule-mask-rate",
        type=float,
        default=None,
        help=(
            "Design masking rate rho of the corruption. When given, also write "
            "an output that fills every test zero whose calibrated detection "
            "probability d = p / (p + rho (1 - p)) exceeds 1/2. The calibration "
            "is cross-fitted on fitting cells; no test label or test score "
            "quantile sets the fill fraction."
        ),
    )
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--condition-column",
        default=None,
        help="Add each gene's mean and zero fraction within the cell's condition (obs column) as selector features.",
    )
    args = parser.parse_args()

    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth_adata = ad.read_h5ad(args.truth)
    counts = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    truth_counts = dense(truth_adata.layers["counts"]).astype(np.float32)
    if counts.shape != truth_adata.shape:
        raise ValueError("truth and corrupted shapes differ")
    if not np.array_equal(
        corrupted_adata.obs_names.astype(str), truth_adata.obs_names.astype(str)
    ):
        raise ValueError("truth and corrupted cell order differ")
    if not np.array_equal(
        corrupted_adata.var_names.astype(str), truth_adata.var_names.astype(str)
    ):
        raise ValueError("truth and corrupted gene order differ")
    splits = pd.read_parquet(args.splits).set_index("cell_id").loc[
        corrupted_adata.obs_names.astype(str), "split"
    ].to_numpy()
    fit = splits == args.fit_split
    test = splits == "test"

    fusion = Path(args.fusion_contract)
    fused_mean = np.load(fusion / "mean.npy", allow_pickle=False)
    teachers = {str(Path(path).name): np.load(Path(path) / "mean.npy", allow_pickle=False) for path in args.teacher_contract}
    feature_names = teacher_feature_names(list(teachers), condition=args.condition_column is not None)

    coordinates = pd.read_parquet(args.coordinates)
    gene_mean = np.log1p(np.mean(counts[fit], axis=0))
    gene_dropout = np.mean(counts[fit] <= 0, axis=0)
    library = np.log1p(counts.sum(axis=1))
    condition_mean = condition_dropout = condition_code = None
    if args.condition_column is not None:
        # Per-condition gene statistics from the selector-fitting cells of each condition.
        labels = corrupted_adata.obs[args.condition_column].astype(str).to_numpy()
        condition_names, condition_code = np.unique(labels, return_inverse=True)
        condition_mean = np.zeros((len(condition_names), counts.shape[1]), dtype=np.float32)
        condition_dropout = np.zeros((len(condition_names), counts.shape[1]), dtype=np.float32)
        for index in range(len(condition_names)):
            members = fit & (condition_code == index)
            if not members.any():
                raise ValueError(f"condition {condition_names[index]} has no selector-fitting cells")
            condition_mean[index] = np.log1p(counts[members].mean(axis=0))
            condition_dropout[index] = (counts[members] <= 0).mean(axis=0)

    def feature_frame(split_mask: np.ndarray, split_name: str, max_rows: int | None = None, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows, cols = np.where((counts == 0) & split_mask[:, None])
        # ``coordinates.split`` records the split used when the corruption was
        # first created.  Cross-fitting supplies a new split file per fold, so
        # that stored label can be stale.  Select masked positives by the
        # current split's cell indices instead.
        split_cells = np.flatnonzero(split_mask)
        split_coordinates = coordinates.loc[
            coordinates["cell_index"].astype(int).isin(split_cells)
        ]
        if max_rows is not None and len(rows) > max_rows:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(len(rows), size=max_rows, replace=False))
            rows, cols = rows[keep], cols[keep]
            positive_rows = split_coordinates["cell_index"].to_numpy(dtype=int)
            positive_cols = split_coordinates["gene_index"].to_numpy(dtype=int)
            combined = {key for key in zip(rows.tolist(), cols.tolist())}
            for r, c in zip(positive_rows, positive_cols):
                combined.add((int(r), int(c)))
            combined = np.asarray(sorted(combined), dtype=np.int64)
            rows, cols = combined[:, 0], combined[:, 1]
        masked_keys = set(zip(split_coordinates["cell_index"].astype(int), split_coordinates["gene_index"].astype(int)))
        labels = np.asarray([1 if (int(r), int(c)) in masked_keys else 0 for r, c in zip(rows, cols)], dtype=np.int8)
        teacher_stack = np.stack([teachers[name][rows, cols] for name in teachers])
        condition_context = None
        if condition_code is not None:
            condition_context = (condition_mean[condition_code[rows], cols], condition_dropout[condition_code[rows], cols])
        features = selector_features(teacher_stack, gene_mean[cols], gene_dropout[cols], library[rows], condition_context)
        return rows, cols, labels, features

    fit_rows, fit_cols, fit_labels, fit_features = feature_frame(fit, args.fit_split, max_rows=args.max_fit_rows, seed=args.seed)
    test_rows, test_cols, test_labels, test_features = feature_frame(test, args.apply_split)

    def candidates(labels: np.ndarray, features: np.ndarray) -> Candidates:
        size = len(labels)
        zeros = np.zeros(size, dtype=np.float32)
        return Candidates(
            rows=np.arange(size, dtype=np.int64),
            cols=np.zeros(size, dtype=np.int64),
            labels=labels,
            features=features,
            fused_values=zeros,
            teacher_values=zeros,
            truth_values=zeros,
            feature_names=feature_names,
        )

    fit_candidates = candidates(fit_labels, fit_features)
    test_candidates = candidates(test_labels, test_features)
    model_reports: dict[str, dict] = {}
    if args.architecture == "rank_ensemble":
        fit_components: list[np.ndarray] = []
        test_components: list[np.ndarray] = []
        for index, architecture in enumerate(ENSEMBLE_ARCHITECTURES):
            component_fit, component_test, component_report = fit_scores(
                "full", architecture, fit_candidates, test_candidates,
                args.max_fit_rows, args.score_batch_rows,
                args.seed + 100 * index,
            )
            model_reports[architecture] = component_report
            fit_components.append(
                score_percentile_from_fit(component_fit, component_fit)
            )
            test_components.append(
                score_percentile_from_fit(component_fit, component_test)
            )
        fit_score = np.mean(fit_components, axis=0).astype(np.float32)
        test_score = np.mean(test_components, axis=0).astype(np.float32)
    else:
        fit_score, test_score, model_report = fit_scores(
            "full", args.architecture, fit_candidates, test_candidates,
            args.max_fit_rows, args.score_batch_rows, args.seed,
        )
        model_reports[args.architecture] = model_report

    if args.curve_points < 2:
        raise ValueError("curve-points must be at least 2")
    if not 0 < args.curve_min_budget < args.curve_max_budget <= 1:
        raise ValueError("curve budgets must satisfy 0 < min < max <= 1")
    ranked_test_labels = test_labels[np.argsort(-test_score, kind="stable")]
    cumulative_true_positives = np.cumsum(ranked_test_labels, dtype=np.int64)
    total_test_positives = int(test_labels.sum())
    exact_budget_curve = []
    for requested_budget in np.linspace(
        args.curve_min_budget, args.curve_max_budget, args.curve_points
    ):
        selected = max(1, int(round(float(requested_budget) * len(test_labels))))
        true_positives = int(cumulative_true_positives[selected - 1])
        precision = true_positives / selected
        recall = (
            true_positives / total_test_positives
            if total_test_positives
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        exact_budget_curve.append(
            {
                "requested_fill_fraction": float(requested_budget),
                "realized_fill_fraction": selected / len(test_labels),
                "n_selected": selected,
                "n_true_positive": true_positives,
                "masked_precision": precision,
                "masked_recall": recall,
                "masked_f1": f1,
            }
        )

    def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
        # Deployment inputs have no masked positives among test zeros.
        if labels.min() == labels.max():
            return {"roc_auc": None, "pr_auc": None}
        return {
            "roc_auc": float(roc_auc_score(labels, scores)),
            "pr_auc": float(average_precision_score(labels, scores)),
        }

    report = {
        args.fit_split: {
            "n_zeros": int(len(fit_labels)),
            "n_masked_positives": int(fit_labels.sum()),
            **ranking_metrics(fit_labels, fit_score),
        },
        "test": {
            "n_zeros": int(len(test_labels)),
            "n_masked_positives": int(test_labels.sum()),
            **ranking_metrics(test_labels, test_score),
        },
        "budgets": {},
        "architecture": args.architecture,
        "budget_mode": args.budget_mode,
        "models": model_reports,
        "exact_budget_curve": exact_budget_curve,
    }

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    fusion_metadata = json.loads((fusion / "metadata.json").read_text())
    cell_ids = corrupted_adata.obs_names.astype(str).tolist()
    gene_ids = corrupted_adata.var_names.astype(str).tolist()
    fit_cells = int(args.fit_cells) if args.fit_cells is not None else int(fit.sum())

    if args.score_gene:
        requested = [str(gene) for gene in args.score_gene]
        gene_lookup = {gene: index for index, gene in enumerate(gene_ids)}
        missing = sorted(set(requested) - set(gene_lookup))
        scored_frames = []
        for split_name, rows, cols, labels, features, scores in [
            (args.fit_split, fit_rows, fit_cols, fit_labels, fit_features, fit_score),
            ("test", test_rows, test_cols, test_labels, test_features, test_score),
        ]:
            keep = np.isin(cols, [gene_lookup[gene] for gene in requested if gene in gene_lookup])
            if not bool(keep.any()):
                continue
            frame = pd.DataFrame(
                {
                    "split": split_name,
                    "cell_index": rows[keep].astype(np.int64),
                    "cell_id": [cell_ids[index] for index in rows[keep]],
                    "gene_index": cols[keep].astype(np.int64),
                    "gene_id": [gene_ids[index] for index in cols[keep]],
                    "selector_score": scores[keep].astype(np.float32),
                    "masked_positive": labels[keep].astype(np.int8),
                }
            )
            for feature_index, feature_name in enumerate(feature_names):
                frame[feature_name] = features[keep, feature_index].astype(np.float32)
            scored_frames.append(frame)
        scored = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()
        scored.to_parquet(output_root / "selected_gene_scores.parquet", index=False)
        (output_root / "selected_gene_scores_metadata.json").write_text(
            json.dumps(
                {
                    "architecture": args.architecture,
                    "fit_split": args.fit_split,
                    "apply_split": args.apply_split,
                    "requested_genes": requested,
                    "missing_genes": missing,
                    "rows_written": int(len(scored)),
                    "test_labels_used_for_training": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    for budget in args.budgets:
        k_fit = max(1, int(round(budget * len(fit_labels))))
        calibration_threshold = float(np.sort(fit_score)[::-1][k_fit - 1])
        if args.budget_mode == "calibration_threshold":
            threshold = calibration_threshold
            fit_fill = fit_score >= threshold
            fill_test = test_score >= threshold
        else:
            fit_fill = exact_topk(fit_score, k_fit)
            fill_test = exact_topk(
                test_score, max(1, int(round(budget * len(test_score))))
            )
            threshold = float(np.min(test_score[fill_test]))
        k_test = int(fill_test.sum())
        output_counts = counts.copy()
        output_counts[test_rows[fill_test], test_cols[fill_test]] = fused_mean[test_rows[fill_test], test_cols[fill_test]]

        fit_recall = float(fit_labels[fit_fill].sum() / fit_labels.sum()) if fit_labels.sum() else 0.0
        fit_false_fill = float((fit_labels[fit_fill] == 0).mean()) if fit_fill.any() else 0.0
        test_recall = float(test_labels[fill_test].sum() / test_labels.sum()) if test_labels.sum() else 0.0
        test_false_fill = float((test_labels[fill_test] == 0).mean()) if fill_test.any() else 0.0
        test_fill_fraction = float(fill_test.mean())
        test_positive = test_labels == 1
        positive_truth = truth_counts[
            test_rows[test_positive], test_cols[test_positive]
        ]
        positive_prediction = np.where(
            fill_test[test_positive],
            np.maximum(
                fused_mean[test_rows[test_positive], test_cols[test_positive]],
                0.0,
            ),
            0.0,
        )
        raw_masked_mae = float(np.log1p(positive_truth).mean()) if test_positive.any() else float("nan")
        method_masked_mae = float(np.abs(
            np.log1p(positive_truth) - np.log1p(positive_prediction)
        ).mean()) if test_positive.any() else float("nan")
        report["budgets"][str(budget)] = {
            "threshold": float(threshold),
            "calibration_threshold": calibration_threshold,
            "threshold_source": (
                args.fit_split
                if args.budget_mode == "calibration_threshold"
                else "unlabelled_apply_score_quantile"
            ),
            f"{args.fit_split}_fill_fraction": float(fit_fill.mean()),
            f"{args.fit_split}_masked_recall": fit_recall,
            f"{args.fit_split}_false_fill_rate": fit_false_fill,
            "test_fill_fraction": test_fill_fraction,
            "test_masked_recall": test_recall,
            "test_false_fill_rate": test_false_fill,
            "test_raw_masked_log1p_mae": raw_masked_mae,
            "test_method_masked_log1p_mae": method_masked_mae,
            "test_masked_error_recovery_fraction": float(
                (raw_masked_mae - method_masked_mae) / raw_masked_mae
            ),
        }

        architecture_suffix = (
            "" if args.architecture == "logistic" else f"_{args.architecture}"
        )
        budget_mode_suffix = "" if args.budget_mode == "calibration_threshold" else "_topk"
        method_name = (
            f"safe_fusion_calibrated{architecture_suffix}{budget_mode_suffix}_"
            f"{str(budget).replace('.', 'p')}"
        )
        metadata = dict(fusion_metadata)
        metadata["method"] = method_name
        metadata["parameters"] = {
            **fusion_metadata.get("parameters", {}),
            "fit_cells": fit_cells,
            "heldout_test_cells": int(test.sum()),
            "test_used_for_fit": False,
            "decision_layer": {
                "calibrator": f"{args.architecture}_on_{args.fit_split}_masked_positives",
                "architecture": args.architecture,
                "features": list(feature_names),
                "budget_mode": args.budget_mode,
                "thresholds_fit_on": (
                    [args.fit_split]
                    if args.budget_mode == "calibration_threshold"
                    else []
                ),
                "test_values_used_for_thresholds": (
                    args.budget_mode == "apply_topk"
                ),
                "test_labels_used_for_thresholds": False,
                "fill_budget": budget,
                "threshold": threshold,
                "calibration_threshold": calibration_threshold,
            },
        }
        write_output_contract(output_root / method_name, output_counts, metadata)

    if args.detection_rule_mask_rate is not None:
        if args.architecture == "rank_ensemble":
            raise ValueError("the detection rule needs a single calibrated selector")
        rho = float(args.detection_rule_mask_rate)
        fit_zero_total = int(((counts == 0) & fit[:, None]).sum())
        coordinate_rows = coordinates["cell_index"].to_numpy(dtype=int)
        coordinate_cols = coordinates["gene_index"].to_numpy(dtype=int)
        fit_positive_total = int(np.sum(fit[coordinate_rows] & (counts[coordinate_rows, coordinate_cols] == 0)))
        frame_negatives = int((fit_labels == 0).sum())
        negative_weight = (fit_zero_total - fit_positive_total) / max(frame_negatives, 1)
        isotonic, out_of_fold, fit_weight = cross_fitted_calibration(
            args.architecture, fit_rows, fit_labels, fit_features, negative_weight,
            args.calibration_folds, args.max_fit_rows, args.score_batch_rows,
            args.seed, candidates,
        )
        fit_probability = isotonic.predict(out_of_fold)
        fit_detection = detection_probability(fit_probability, rho)
        test_probability = isotonic.predict(test_score)
        test_detection = detection_probability(test_probability, rho)
        fill_rule = test_detection > 0.5
        n_rule = int(fill_rule.sum())
        rule_tp = int(test_labels[fill_rule].sum())
        rule_precision = rule_tp / n_rule if n_rule else 0.0
        rule_recall = rule_tp / total_test_positives if total_test_positives else 0.0
        rule_f1 = (
            2 * rule_precision * rule_recall / (rule_precision + rule_recall)
            if rule_precision + rule_recall else 0.0
        )
        quantiles = [0.5, 0.75, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
        rule_report = {
            "mask_rate": rho,
            "probability_threshold": rho / (1.0 + rho),
            "calibration": "isotonic on cross-fitted fitting-cell scores",
            "calibration_folds": int(args.calibration_folds),
            "fit_candidates_total": fit_zero_total,
            "fit_positives_total": fit_positive_total,
            "fit_negative_weight": float(negative_weight),
            "fit_fill_fraction": float(np.average(fit_detection > 0.5, weights=fit_weight)),
            "fit_expected_positive_share": float(np.average(fit_probability, weights=fit_weight)),
            "fit_observed_positive_share": float(np.average(fit_labels, weights=fit_weight)),
            "fit_out_of_fold_reliability": reliability_table(fit_probability, fit_labels, fit_weight),
            "isotonic_map": [
                {"score_quantile": q, "score": float(s), "probability": float(isotonic.predict([s])[0])}
                for q, s in zip(quantiles, np.quantile(out_of_fold, quantiles))
            ],
            "test_fill_fraction": float(fill_rule.mean()),
            "test_n_selected": n_rule,
            "test_expected_positive_share": float(test_probability.mean()),
            "test_labels_used": False,
            "test_score_quantiles_used": False,
        }
        if total_test_positives:
            best = max(exact_budget_curve, key=lambda row: row["masked_f1"])
            rule_report.update({
                "test_masked_precision": rule_precision,
                "test_masked_recall": rule_recall,
                "test_masked_f1": rule_f1,
                "test_observed_positive_share": float(test_labels.mean()),
                "test_reliability": reliability_table(test_probability, test_labels.astype(np.float64)),
                "best_fill_fraction_in_hindsight": best["realized_fill_fraction"],
                "best_masked_f1_in_hindsight": best["masked_f1"],
            })
        report["detection_rule"] = rule_report
        output_counts = counts.copy()
        output_counts[test_rows[fill_rule], test_cols[fill_rule]] = fused_mean[test_rows[fill_rule], test_cols[fill_rule]]
        method_name = f"safe_fusion_calibrated_{args.architecture}_detection_rule"
        metadata = dict(fusion_metadata)
        metadata["method"] = method_name
        metadata["parameters"] = {
            **fusion_metadata.get("parameters", {}),
            "fit_cells": fit_cells,
            "heldout_test_cells": int(test.sum()),
            "test_used_for_fit": False,
            "decision_layer": {
                "calibrator": f"{args.architecture}_on_{args.fit_split}_masked_positives",
                "architecture": args.architecture,
                "features": list(feature_names),
                "budget_mode": "detection_rule",
                "rule": "fill when p / (p + rho (1 - p)) > 1/2",
                "mask_rate": rho,
                "test_values_used_for_thresholds": False,
                "test_labels_used_for_thresholds": False,
                "fill_fraction": float(fill_rule.mean()),
            },
        }
        write_output_contract(output_root / method_name, output_counts, metadata)

    (output_root / "calibration_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
