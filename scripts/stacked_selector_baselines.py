#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import (
    CURVE_BUDGETS,
    EVIDENCE,
    NATIVE_SCALES,
    STACKED_COMPARATORS,
    UNIT_FRACTIONS,
    add_root_arguments,
    count_scale_values,
    load_unit,
    unit_counts,
    units_from_args,
)
from selector_attribution import (
    Candidates,
    exact_topk,
    fit_scores,
    selector_features,
    teacher_feature_names,
)


def candidate_keys(counts: np.ndarray, split_mask: np.ndarray, masked: np.ndarray, max_rows: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where((counts == 0) & split_mask[:, None])
    if max_rows is not None and len(rows) > max_rows:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(rows), size=max_rows, replace=False))
        n_genes = counts.shape[1]
        positive_rows, positive_cols = np.where(masked & split_mask[:, None])
        keys = np.union1d(
            rows[keep].astype(np.int64) * n_genes + cols[keep],
            positive_rows.astype(np.int64) * n_genes + positive_cols,
        )
        rows, cols = keys // n_genes, keys % n_genes
    return rows, cols


def exact_budget_curve(scores: np.ndarray, labels: np.ndarray) -> list[dict[str, float]]:
    cumulative = np.cumsum(labels[np.argsort(-scores, kind="stable")], dtype=np.int64)
    positives = int(labels.sum())
    curve = []
    for budget in CURVE_BUDGETS:
        selected = max(1, int(round(float(budget) * len(labels))))
        true_positive = int(cumulative[selected - 1])
        precision = true_positive / selected
        recall = true_positive / positives if positives else 0.0
        curve.append({
            "requested_fill_fraction": float(budget),
            "realized_fill_fraction": selected / len(labels),
            "n_selected": selected,
            "n_true_positive": true_positive,
            "masked_precision": precision,
            "masked_recall": recall,
            "masked_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        })
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    add_root_arguments(parser)
    parser.add_argument("--unit", required=True, help="pancreas_0, pancreas_1, pancreas_2, colon or norman_crispra")
    parser.add_argument("--output-root", type=Path, default=EVIDENCE / "stacked_selector_baselines")
    parser.add_argument("--comparators", nargs="+", default=list(STACKED_COMPARATORS))
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    units = {unit.key: unit for unit in units_from_args(args, args.seed)}
    unit = units[args.unit]
    data = load_unit(unit)
    fit_mask = data.split == unit.fit_split
    test_mask = data.split == "test"
    counts = data.counts
    gene_mean = np.log1p(np.mean(counts[fit_mask], axis=0))
    gene_dropout = np.mean(counts[fit_mask] <= 0, axis=0)
    log_library = np.log1p(data.library)

    fit_rows, fit_cols = candidate_keys(counts, fit_mask, data.masked, args.max_fit_rows, args.seed)
    test_rows, test_cols = np.where((counts == 0) & test_mask[:, None])
    fit_labels = data.masked[fit_rows, fit_cols].astype(np.int8)
    test_labels = data.masked[test_rows, test_cols].astype(np.int8)
    test_units = data.unit_labels[test_rows]

    def candidates(name: str, rows: np.ndarray, cols: np.ndarray, labels: np.ndarray, values: np.ndarray) -> Candidates:
        placeholder = np.zeros(len(rows), dtype=np.float32)
        return Candidates(
            rows=rows, cols=cols, labels=labels,
            features=selector_features(values[None, :], gene_mean[cols], gene_dropout[cols], log_library[rows]),
            fused_values=placeholder, teacher_values=values[None, :], truth_values=placeholder,
            feature_names=teacher_feature_names([name]),
        )

    for name in args.comparators:
        started = time.perf_counter()
        contract = unit.contracts[name]
        fit_values, scale = count_scale_values(contract, data, fit_rows, fit_cols)
        test_values, _ = count_scale_values(contract, data, test_rows, test_cols)
        if scale in NATIVE_SCALES:
            offset = float(np.min(fit_values))
            fit_values = fit_values - offset
            test_values = test_values - offset
        fit_score, test_score, model_report = fit_scores(
            "full", "mlp",
            candidates(name, fit_rows, fit_cols, fit_labels, fit_values),
            candidates(name, test_rows, test_cols, test_labels, test_values),
            args.max_fit_rows, args.score_batch_rows, args.seed,
        )
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        output = args.output_root / unit.key / slug
        output.mkdir(parents=True, exist_ok=True)
        frames = []
        for fraction in UNIT_FRACTIONS:
            selected = exact_topk(test_score, max(1, int(round(fraction * len(test_score)))))
            frame = unit_counts(selected, test_labels, test_units)
            frame.insert(0, "fraction", fraction)
            frame.insert(0, "method", f"{name} (stacked)")
            frame.insert(0, "group", unit.key)
            frame.insert(0, "dataset", unit.dataset)
            frames.append(frame)
        pd.concat(frames, ignore_index=True).to_parquet(output / "unit_counts.parquet", index=False)
        report = {
            "comparator": name,
            "contract": str(contract),
            "stored_scale": scale,
            "ranking_value": "native score" if scale in NATIVE_SCALES else "count scale of the masked input",
            "dataset": unit.dataset,
            "unit": unit.key,
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
            "test_labels_used_for_training": False,
            "model": model_report,
            "elapsed_seconds": time.perf_counter() - started,
            "exact_budget_curve": exact_budget_curve(test_score, test_labels),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps({
            "unit": unit.key, "comparator": name, "test_pr_auc": report["test"]["pr_auc"],
            "elapsed_seconds": round(report["elapsed_seconds"], 1),
        }), flush=True)


if __name__ == "__main__":
    main()
