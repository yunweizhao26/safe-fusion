#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import count_scale_values, load_unit
from selector_attribution import (
    balanced_sample_weight,
    predict_scores_in_batches,
    selector_features,
    stratified_fit_indices,
)
from stacked_selector_baselines import candidate_keys, exact_budget_curve
from v2_ablation_common import PRODUCTION_HIDDEN, TEACHERS, UNIT_KEYS, production_teacher_root, units

def fit_mlp(hidden: tuple[int, ...], features: np.ndarray, labels: np.ndarray, columns: np.ndarray,
            max_fit_rows: int, seed: int):

    fit_index = stratified_fit_indices(labels, max_fit_rows, seed)
    train_features = features[fit_index][:, columns]
    train_labels = labels[fit_index]
    sample_weight = balanced_sample_weight(train_labels)
    scaler = StandardScaler().fit(train_features)
    model = MLPClassifier(
        hidden_layer_sizes=hidden,
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
    model.fit(scaler.transform(train_features), train_labels, sample_weight=sample_weight)
    return model, scaler, int(len(fit_index))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", choices=UNIT_KEYS, required=True)
    parser.add_argument("--hidden", type=int, nargs="+", default=list(PRODUCTION_HIDDEN))
    parser.add_argument("--teacher-root", type=Path, default=None,
                        help="Directory with the five teacher contracts (default: the unit's production teachers).")
    parser.add_argument("--gene-median-contract", type=Path, default=None,
                        help="Gene median contract when it is not under --teacher-root.")
    parser.add_argument("--corrupted", type=Path, default=None, help="Masked input (default: the benchmark input).")
    parser.add_argument("--coordinates", type=Path, default=None, help="Masked coordinates of --corrupted.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    started = time.perf_counter()

    unit = units()[args.unit]
    if args.corrupted is not None:
        unit = replace(unit, corrupted=args.corrupted, coordinates=args.coordinates)
    teacher_root = args.teacher_root or production_teacher_root(unit)
    contracts = [teacher_root / name for name in TEACHERS]
    if args.gene_median_contract is not None:
        contracts[0] = args.gene_median_contract

    data = load_unit(unit)
    counts = data.counts
    fit_mask = data.split == unit.fit_split
    test_mask = data.split == "test"
    gene_mean = np.log1p(np.mean(counts[fit_mask], axis=0))
    gene_dropout = np.mean(counts[fit_mask] <= 0, axis=0)
    log_library = np.log1p(counts.sum(axis=1))
    fit_rows, fit_cols = candidate_keys(counts, fit_mask, data.masked, args.max_fit_rows, args.seed)
    test_rows, test_cols = np.where((counts == 0) & test_mask[:, None])
    fit_labels = data.masked[fit_rows, fit_cols].astype(np.int8)
    test_labels = data.masked[test_rows, test_cols].astype(np.int8)

    def features(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        stack = np.stack([count_scale_values(contract, data, rows, cols)[0] for contract in contracts])
        return selector_features(stack, gene_mean[cols], gene_dropout[cols], log_library[rows])

    fit_features = features(fit_rows, fit_cols)
    test_features = features(test_rows, test_cols)
    hidden = tuple(args.hidden)
    fit_started = time.perf_counter()
    columns = np.arange(fit_features.shape[1])
    model, scaler, fit_rows_used = fit_mlp(hidden, fit_features, fit_labels, columns, args.max_fit_rows, args.seed)
    fit_seconds = time.perf_counter() - fit_started
    fit_score = predict_scores_in_batches(model, scaler, fit_features, columns, args.score_batch_rows)
    test_score = predict_scores_in_batches(model, scaler, test_features, columns, args.score_batch_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "test_scores.npy", test_score.astype(np.float32), allow_pickle=False)
    report = {
        "unit": unit.key,
        "dataset": unit.dataset,
        "hidden_layer_sizes": list(hidden),
        "corrupted": str(unit.corrupted),
        "coordinates": str(unit.coordinates),
        "contracts": [str(path) for path in contracts],
        "fit_split": unit.fit_split,
        unit.fit_split: {
            "n_zeros": int(len(fit_labels)),
            "n_masked_positives": int(fit_labels.sum()),
            "roc_auc": float(roc_auc_score(fit_labels, fit_score)),
            "pr_auc": float(average_precision_score(fit_labels, fit_score)),
        },
        "test": {
            "n_zeros": int(len(test_labels)),
            "n_masked_positives": int(test_labels.sum()),
            "roc_auc": float(roc_auc_score(test_labels, test_score)),
            "pr_auc": float(average_precision_score(test_labels, test_score)),
        },
        "test_candidate_order": "row-major (cell, gene) order of the zeros of the held-out cells",
        "test_labels_used_for_training": False,
        "seed": args.seed,
        "model": {"fit_rows": fit_rows_used, "fit_seconds": fit_seconds, "model_iterations": int(model.n_iter_),
                  "model_parameters": model.get_params(deep=False)},
        "elapsed_seconds": time.perf_counter() - started,
        "exact_budget_curve": exact_budget_curve(test_score, test_labels),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"unit": unit.key, "hidden": list(hidden), "test_pr_auc": round(report["test"]["pr_auc"], 5),
                      "iterations": int(model.n_iter_), "elapsed_seconds": round(report["elapsed_seconds"], 1)}), flush=True)

if __name__ == "__main__":
    main()
