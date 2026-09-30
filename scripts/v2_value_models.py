#!/usr/bin/env python3

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
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.base import clone

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from calibrated_selective_fill import cross_fitted_calibration, detection_probability
from run_leakage_safe_method import MAX_VALUE_FIT_ENTRIES, fit_value_model, predict_value_model, value_features
from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file
from safefusion_benchmark.splits import FOLDS, training_folds
from selector_attribution import Candidates, fit_scores, selector_features, teacher_feature_names
from stacked_selector_baselines import candidate_keys

TEACHERS = ("gene_median", "svd_impute", "graph_smooth", "magic_inductive", "scvi_inductive")
VALUES = (
    "conditional",
    "rate_poisson",
    "rate_negbin",
    "conditional_detection",
    "poisson_all",
    "poisson_all_unweighted",
)
TRAINING_SPLITS = ("development", "validation")

SELECTOR_ARCHITECTURE = "mlp"
MAX_SELECTOR_ROWS = 2_000_000
SCORE_BATCH_ROWS = 250_000
CALIBRATION_FOLDS = 5

INVERSION_STEPS = 80

THETA_RANGE = (1e-3, 1e6)
PREDICT_BATCH = 2_000_000

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

@dataclass
class Benchmark:
    counts: np.ndarray
    obs: pd.DataFrame
    split: np.ndarray
    hidden: np.ndarray
    masked: np.ndarray
    teachers: dict[str, np.ndarray]
    positive_rows: np.ndarray
    positive_cols: np.ndarray
    cell_ids: list[str]
    gene_ids: list[str]
    paths: dict[str, str]

    @property
    def training(self) -> np.ndarray:
        return np.isin(self.split, TRAINING_SPLITS)

def load_benchmark(input_path: Path, coordinates_path: Path, splits_path: Path, teacher_root: Path,
                   positive_splits: tuple[str, ...] = TRAINING_SPLITS) -> Benchmark:

    adata = ad.read_h5ad(input_path)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = dense(matrix).astype(np.float32)
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    split = pd.read_parquet(splits_path).set_index("cell_id").loc[cell_ids, "split"].astype(str).to_numpy()
    coordinates = pd.read_parquet(coordinates_path)
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    keep = np.isin(split[rows], positive_splits)
    rows, cols = rows[keep], cols[keep]
    if np.any(counts[rows, cols] != 0):
        raise ValueError(f"masked coordinates of {coordinates_path} are not zero in the input")
    hidden = np.zeros(counts.shape, dtype=np.float32)
    hidden[rows, cols] = coordinates["original_value"].to_numpy(dtype=np.float32)[keep]
    if np.any(hidden[rows, cols] <= 0):
        raise ValueError("a masked positive has no positive hidden count")
    teachers = {}
    for name in TEACHERS:
        contract = teacher_root / name
        metadata = json.loads((contract / "metadata.json").read_text())
        if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids or metadata.get("scale") != "counts":
            raise ValueError(f"teacher {contract} does not match the input or is not on the count scale")
        teachers[name] = np.maximum(np.load(contract / "mean.npy"), 0.0).astype(np.float32)
    return Benchmark(
        counts=counts, obs=adata.obs.copy(), split=split, hidden=hidden, masked=hidden > 0,
        teachers=teachers, positive_rows=rows, positive_cols=cols, cell_ids=cell_ids, gene_ids=gene_ids,
        paths={"input": str(input_path), "coordinates": str(coordinates_path), "splits": str(splits_path),
               "teacher_root": str(teacher_root)},
    )

@dataclass
class ValueContext:
    gene_mean: np.ndarray
    detection: np.ndarray
    library_log: np.ndarray

def value_context(bench: Benchmark, fitting: np.ndarray) -> ValueContext:

    fit_counts = bench.counts[fitting]
    return ValueContext(
        gene_mean=np.log1p(fit_counts.mean(axis=0)).astype(np.float32),
        detection=(fit_counts > 0).mean(axis=0).astype(np.float32),
        library_log=np.log1p(bench.counts.sum(axis=1)).astype(np.float32),
    )

def value_inputs(bench: Benchmark, context: ValueContext, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    teacher_logs = np.column_stack([np.log1p(bench.teachers[name][rows, cols]) for name in TEACHERS]).astype(np.float32)
    features = value_features(teacher_logs, context.gene_mean, context.detection, context.library_log, rows, cols)
    return teacher_logs, features

def truncated_poisson_rate(mean: np.ndarray) -> np.ndarray:

    mean = np.asarray(mean, dtype=np.float64)
    result = np.zeros_like(mean)
    above = mean > 1.0
    target = mean[above]
    low, high = target - 1.0, target.copy()
    for _ in range(INVERSION_STEPS):
        middle = 0.5 * (low + high)
        value = middle / -np.expm1(-middle)
        below = value < target
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    result[above] = 0.5 * (low + high)
    return result

def negbin_zero_log(mu: np.ndarray, theta: float) -> np.ndarray:
    return -theta * np.log1p(mu / theta)

def truncated_negbin_mean(mean: np.ndarray, theta: float) -> np.ndarray:

    mean = np.asarray(mean, dtype=np.float64)
    result = np.zeros_like(mean)
    above = mean > 1.0
    target = mean[above]
    low, high = np.zeros_like(target), target.copy()
    for _ in range(INVERSION_STEPS):
        middle = 0.5 * (low + high)
        value = middle / -np.expm1(negbin_zero_log(middle, theta))
        below = value < target
        low = np.where(below, middle, low)
        high = np.where(below, high, middle)
    result[above] = 0.5 * (low + high)
    return result

def truncated_negbin_loglik(y: np.ndarray, mu: np.ndarray, theta: float) -> np.ndarray:
    log_zero = negbin_zero_log(mu, theta)
    return (
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1.0)
        + log_zero + y * (np.log(mu) - np.log(theta + mu))
        - np.log(-np.expm1(log_zero))
    )

def fit_dispersion(y: np.ndarray, conditional_mean: np.ndarray) -> float:

    informative = conditional_mean > 1.0
    y, conditional_mean = y[informative].astype(np.float64), conditional_mean[informative]

    def negative(log_theta: float) -> float:
        theta = float(np.exp(log_theta))
        mu = truncated_negbin_mean(conditional_mean, theta)
        return -float(truncated_negbin_loglik(y, mu, theta).sum())

    result = minimize_scalar(negative, bounds=tuple(np.log(THETA_RANGE)), method="bounded")
    return float(np.exp(result.x))

def positive_entries(bench: Benchmark, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    keep = cells[bench.positive_rows]
    rows, cols = bench.positive_rows[keep], bench.positive_cols[keep]
    return rows, cols, bench.hidden[rows, cols].astype(np.float64)

def subsample(n: int, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=min(n, MAX_VALUE_FIT_ENTRIES), replace=False))

@dataclass
class FittedValues:
    conditional: tuple
    count_mean: object
    theta: float
    all_weighted: object
    all_unweighted: object
    context: ValueContext
    report: dict

def conditional_offset(conditional: tuple, teacher_logs: np.ndarray, features: np.ndarray) -> np.ndarray:

    return np.exp(np.asarray(predict_value_model(conditional, teacher_logs, features), dtype=np.float64))

def fit_poisson_offset(template, features: np.ndarray, target: np.ndarray, offset: np.ndarray,
                       weight: np.ndarray | None = None):

    weight = offset if weight is None else weight * offset
    return clone(template).set_params(loss="poisson").fit(features, target / offset, sample_weight=weight)

def fit_values(bench: Benchmark, fitting: np.ndarray, mask_rate: float, seed: int) -> FittedValues:

    started = time.perf_counter()
    context = value_context(bench, fitting)
    rows, cols, y = positive_entries(bench, fitting)
    teacher_logs, features = value_inputs(bench, context, rows, cols)
    conditional = fit_value_model("boosted", teacher_logs, features, np.log1p(y), seed)
    template = conditional[1]
    keep = subsample(len(y), seed)
    kept_logs, kept_features, kept_y = teacher_logs[keep], features[keep], y[keep]

    count_mean = fit_poisson_offset(template, kept_features, kept_y, conditional_offset(conditional, kept_logs, kept_features))

    fitting_rows = np.flatnonzero(fitting)
    fold_of_row = np.full(len(bench.counts), -1)
    fold_of_row[fitting_rows] = training_folds(len(fitting_rows), seed)
    kept_fold = fold_of_row[rows[keep]]
    out_of_fold = np.empty(len(keep))
    for fold in range(FOLDS):
        held = kept_fold == fold
        fold_conditional = fit_value_model("boosted", kept_logs[~held], kept_features[~held], np.log1p(kept_y[~held]), seed)
        model = fit_poisson_offset(template, kept_features[~held], kept_y[~held],
                                   conditional_offset(fold_conditional, kept_logs[~held], kept_features[~held]))
        out_of_fold[held] = conditional_offset(fold_conditional, kept_logs[held], kept_features[held]) * model.predict(kept_features[held])
    theta = fit_dispersion(kept_y, out_of_fold)

    candidate_rows, candidate_cols = candidate_keys(bench.counts, fitting, bench.masked, MAX_SELECTOR_ROWS, seed)
    labels = bench.masked[candidate_rows, candidate_cols]
    target = bench.hidden[candidate_rows, candidate_cols].astype(np.float64)
    candidate_logs, candidate_features = value_inputs(bench, context, candidate_rows, candidate_cols)
    candidate_offset = conditional_offset(conditional, candidate_logs, candidate_features)
    zero_total = int(((bench.counts == 0) & fitting[:, None]).sum()) - len(y)
    zero_weight = zero_total / max(int((~labels).sum()), 1)
    weighted = np.where(labels, 1.0 / mask_rate, zero_weight)
    unweighted = np.where(labels, 1.0, zero_weight)
    all_weighted = fit_poisson_offset(template, candidate_features, target, candidate_offset, weighted)
    all_unweighted = fit_poisson_offset(template, candidate_features, target, candidate_offset, unweighted)

    report = {
        "value_fitting_cells": int(fitting.sum()),
        "masked_positives": int(len(y)),
        "masked_positives_fitted": int(len(keep)),
        "conditional_iterations": int(template.n_iter_),
        "count_mean_iterations": int(count_mean.n_iter_),
        "negative_binomial_dispersion": theta,
        "candidates_fitted": int(len(labels)),
        "candidate_positives_fitted": int(labels.sum()),
        "recorded_zero_weight": float(zero_weight),
        "mask_rate": float(mask_rate),
        "poisson_all_iterations": int(all_weighted.n_iter_),
        "poisson_all_unweighted_iterations": int(all_unweighted.n_iter_),
        "poisson_offset": "exp of the conditional log value",
        "boosting_parameters": {key: value for key, value in template.get_params().items() if key != "loss"},
        "fit_seconds": time.perf_counter() - started,
    }
    return FittedValues(conditional, count_mean, theta, all_weighted, all_unweighted, context, report)

def predict_values(fitted: FittedValues, bench: Benchmark, rows: np.ndarray, cols: np.ndarray,
                   detection: np.ndarray | None = None) -> dict[str, np.ndarray]:

    parts: dict[str, list[np.ndarray]] = {name: [] for name in VALUES if name != "conditional_detection"}
    for start in range(0, len(rows), PREDICT_BATCH):
        block_rows, block_cols = rows[start:start + PREDICT_BATCH], cols[start:start + PREDICT_BATCH]
        teacher_logs, features = value_inputs(bench, fitted.context, block_rows, block_cols)
        log_value = np.asarray(predict_value_model(fitted.conditional, teacher_logs, features), dtype=np.float64)
        offset = np.exp(log_value)

        parts["conditional"].append(np.expm1(np.maximum(log_value.astype(np.float32), 0.0)))
        conditional_mean = offset * fitted.count_mean.predict(features)
        parts["rate_poisson"].append(truncated_poisson_rate(conditional_mean))
        parts["rate_negbin"].append(truncated_negbin_mean(conditional_mean, fitted.theta))
        parts["poisson_all"].append(offset * fitted.all_weighted.predict(features))
        parts["poisson_all_unweighted"].append(offset * fitted.all_unweighted.predict(features))
    values = {name: (np.concatenate(chunks) if chunks else np.zeros(0)).astype(np.float32) for name, chunks in parts.items()}
    if detection is not None:
        values["conditional_detection"] = (values["conditional"] * detection).astype(np.float32)
    return values

def selector_candidates(bench: Benchmark, gene_mean: np.ndarray, gene_dropout: np.ndarray, library: np.ndarray,
                        rows: np.ndarray, cols: np.ndarray) -> Candidates:

    stack = np.stack([bench.teachers[name][rows, cols] for name in TEACHERS])
    labels = bench.masked[rows, cols].astype(np.int8)
    placeholder = np.zeros(len(rows), dtype=np.float32)
    return Candidates(
        rows=rows, cols=cols, labels=labels,
        features=selector_features(stack, gene_mean[cols], gene_dropout[cols], library[rows]),
        fused_values=placeholder, teacher_values=placeholder, truth_values=placeholder,
        feature_names=teacher_feature_names(list(TEACHERS)),
    )

def selector_detection(bench: Benchmark, selector_cells: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                       mask_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:

    started = time.perf_counter()
    fit_counts = bench.counts[selector_cells]
    gene_mean = np.log1p(np.mean(fit_counts, axis=0))
    gene_dropout = np.mean(fit_counts <= 0, axis=0)
    library = np.log1p(bench.counts.sum(axis=1))
    fit_rows, fit_cols = candidate_keys(bench.counts, selector_cells, bench.masked, MAX_SELECTOR_ROWS, seed)
    fit = selector_candidates(bench, gene_mean, gene_dropout, library, fit_rows, fit_cols)
    target = selector_candidates(bench, gene_mean, gene_dropout, library, rows, cols)
    _, score, model_report = fit_scores("full", SELECTOR_ARCHITECTURE, fit, target, MAX_SELECTOR_ROWS, SCORE_BATCH_ROWS, seed)
    zero_total = int(((bench.counts == 0) & selector_cells[:, None]).sum())
    positive_total = int((bench.masked & selector_cells[:, None]).sum())
    negative_weight = (zero_total - positive_total) / max(int((fit.labels == 0).sum()), 1)

    def make_candidates(labels: np.ndarray, features: np.ndarray) -> Candidates:
        placeholder = np.zeros(len(labels), dtype=np.float32)
        return Candidates(
            rows=np.arange(len(labels)), cols=np.zeros(len(labels), dtype=np.int64), labels=labels,
            features=features, fused_values=placeholder, teacher_values=placeholder, truth_values=placeholder,
            feature_names=fit.feature_names,
        )

    isotonic, _, _ = cross_fitted_calibration(
        SELECTOR_ARCHITECTURE, fit_rows, fit.labels, fit.features, negative_weight, CALIBRATION_FOLDS,
        MAX_SELECTOR_ROWS, SCORE_BATCH_ROWS, seed, make_candidates,
    )
    probability = isotonic.predict(score).astype(np.float32)
    detection = detection_probability(probability, mask_rate).astype(np.float32)
    report = {
        "selector_cells": int(selector_cells.sum()),
        "selector_fit_rows": int(model_report["fit_rows"]),
        "selector_iterations": model_report["model_iterations"],
        "calibration_folds": CALIBRATION_FOLDS,
        "negative_weight": float(negative_weight),
        "seconds": time.perf_counter() - started,
    }
    return score, probability, detection, report

def write_value_contracts(bench: Benchmark, fitted: FittedValues, values: list[str], detection_rows: np.ndarray | None,
                          detection_cols: np.ndarray | None, detection: np.ndarray | None, output_root: Path, seed: int,
                          extra: dict) -> None:

    n_cells, n_genes = bench.counts.shape
    predicted = [name for name in VALUES if name != "conditional_detection" and (name in values or name == "conditional")]
    matrices = {name: np.zeros((n_cells, n_genes), dtype=np.float32) for name in predicted}
    all_cols = np.arange(n_genes)
    block = max(1, PREDICT_BATCH // n_genes)
    for start in range(0, n_cells, block):
        cells = np.arange(start, min(start + block, n_cells))
        rows, cols = np.repeat(cells, n_genes), np.tile(all_cols, len(cells))
        block_values = predict_values(fitted, bench, rows, cols)
        for name in predicted:
            matrices[name][cells] = block_values[name].reshape(len(cells), n_genes)
    if "conditional_detection" in values:
        matrices["conditional_detection"] = matrices["conditional"].copy()
        matrices["conditional_detection"][detection_rows, detection_cols] *= detection
    for name in values:
        metadata = {
            "method": f"safe_fusion_value_{name}",
            "scale": "counts",
            "cell_ids": bench.cell_ids,
            "gene_ids": bench.gene_ids,
            "cell_order_sha256": order_hash(bench.cell_ids),
            "gene_order_sha256": order_hash(bench.gene_ids),
            "training_splits": list(TRAINING_SPLITS),
            "parameters": {
                "value": name,
                "value_models": fitted.report,
                "detection_entries": "candidate zeros of the target cells" if name == "conditional_detection" else None,
                "test_used_for_fit": False,
                **extra,
            },
            "seed": seed,
            "input_sha256": sha256_file(bench.paths["input"]),
            "coordinates_sha256": sha256_file(bench.paths["coordinates"]),
        }
        write_output_contract(output_root / name, matrices[name], metadata)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True, help="Directory holding the five teacher contracts.")
    parser.add_argument("--fit-split", choices=list(TRAINING_SPLITS), required=True, help="Selector-fitting split.")
    parser.add_argument("--target-split", default="test", help="Cells whose candidate zeros receive a detection probability.")
    parser.add_argument("--values", nargs="+", choices=VALUES, default=list(VALUES))
    parser.add_argument("--mask-rate", type=float, default=0.10, help="Design masking rate of the fitting cells.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    bench = load_benchmark(args.input, args.coordinates, args.splits, args.teacher_root)
    fitted = fit_values(bench, bench.training, args.mask_rate, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = cols = detection = None
    detection_report = None
    if "conditional_detection" in args.values:
        target = bench.split == args.target_split
        rows, cols = np.nonzero((bench.counts == 0) & target[:, None])
        score, probability, detection, detection_report = selector_detection(
            bench, bench.split == args.fit_split, rows, cols, args.mask_rate, args.seed,
        )
        np.savez(
            args.output_root / "detection.npz", rows=rows.astype(np.int32), cols=cols.astype(np.int32),
            score=score, probability=probability, detection=detection,
        )
    write_value_contracts(bench, fitted, list(args.values), rows, cols, detection, args.output_root, args.seed, {
        "fit_split": args.fit_split, "target_split": args.target_split, "detection": detection_report,
    })
    report = {"values": list(args.values), "value_models": fitted.report, "detection": detection_report, "paths": bench.paths}
    (args.output_root / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
