#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import NATIVE_SCALES, UnitData, count_scale_values, dense
from selector_attribution import Candidates, fit_scores, selector_features, teacher_feature_names
from stacked_selector_baselines import candidate_keys, exact_budget_curve


def load_input(corrupted: Path, coordinates: Path, splits: Path, unit_column: str) -> UnitData:
    adata = ad.read_h5ad(corrupted)
    counts = dense(adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(splits).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    frame = pd.read_parquet(coordinates)
    labelled = np.zeros(counts.shape, dtype=bool)
    labelled[frame["cell_index"].to_numpy(dtype=int), frame["gene_index"].to_numpy(dtype=int)] = True
    return UnitData(
        counts=counts,
        split=split,
        unit_labels=adata.obs[unit_column].astype(str).to_numpy(),
        library=counts.sum(axis=1, dtype=np.float64),
        masked=labelled,
        cell_ids=adata.obs_names.astype(str).tolist(),
        gene_ids=adata.var_names.astype(str).tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corrupted", type=Path, required=True, help="model input, counts in layers['corrupted_counts']")
    parser.add_argument("--coordinates", type=Path, required=True, help="labelled zeros (masked or thinning positives)")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fit-split", required=True, choices=["development", "validation"])
    parser.add_argument("--unit-column", required=True, help="obs column of the biological unit, stored with the scores")
    parser.add_argument("--contract", type=Path, required=True, help="the method's output contract")
    parser.add_argument("--name", required=True, help="method name used as the teacher feature name")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    started = time.perf_counter()
    data = load_input(args.corrupted, args.coordinates, args.splits, args.unit_column)
    fit_mask = data.split == args.fit_split
    test_mask = data.split == "test"
    counts = data.counts
    gene_mean = np.log1p(np.mean(counts[fit_mask], axis=0))
    gene_dropout = np.mean(counts[fit_mask] <= 0, axis=0)
    log_library = np.log1p(data.library)

    fit_rows, fit_cols = candidate_keys(counts, fit_mask, data.masked, args.max_fit_rows, args.seed)
    test_rows, test_cols = np.where((counts == 0) & test_mask[:, None])
    fit_labels = data.masked[fit_rows, fit_cols].astype(np.int8)
    test_labels = data.masked[test_rows, test_cols].astype(np.int8)

    fit_values, scale = count_scale_values(args.contract, data, fit_rows, fit_cols)
    test_values, _ = count_scale_values(args.contract, data, test_rows, test_cols)
    if scale in NATIVE_SCALES:
        offset = float(np.min(fit_values))
        fit_values, test_values = fit_values - offset, test_values - offset

    def candidates(rows: np.ndarray, cols: np.ndarray, labels: np.ndarray, values: np.ndarray) -> Candidates:
        placeholder = np.zeros(len(rows), dtype=np.float32)
        return Candidates(
            rows=rows, cols=cols, labels=labels,
            features=selector_features(values[None, :], gene_mean[cols], gene_dropout[cols], log_library[rows]),
            fused_values=placeholder, teacher_values=values[None, :], truth_values=placeholder,
            feature_names=teacher_feature_names([args.name]),
        )

    fit_score, test_score, model_report = fit_scores(
        "full", "mlp",
        candidates(fit_rows, fit_cols, fit_labels, fit_values),
        candidates(test_rows, test_cols, test_labels, test_values),
        args.max_fit_rows, args.score_batch_rows, args.seed,
    )

    def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
        if labels.min() == labels.max():
            return {"roc_auc": None, "pr_auc": None}
        return {"roc_auc": float(roc_auc_score(labels, scores)), "pr_auc": float(average_precision_score(labels, scores))}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "test_scores.npz",
        rows=test_rows.astype(np.int64), cols=test_cols.astype(np.int64),
        labels=test_labels, score=test_score.astype(np.float32),
        units=data.unit_labels[test_rows].astype(str),
    )
    report = {
        "method": args.name,
        "contract": str(args.contract),
        "stored_scale": scale,
        "corrupted": str(args.corrupted),
        "coordinates": str(args.coordinates),
        "splits": str(args.splits),
        "fit_split": args.fit_split,
        args.fit_split: {"n_zeros": int(len(fit_labels)), "n_labelled": int(fit_labels.sum()), **metrics(fit_labels, fit_score)},
        "test": {"n_zeros": int(len(test_labels)), "n_labelled": int(test_labels.sum()), **metrics(test_labels, test_score)},
        "test_labels_used_for_training": False,
        "model": model_report,
        "elapsed_seconds": time.perf_counter() - started,
        "exact_budget_curve": exact_budget_curve(test_score, test_labels) if test_labels.any() else [],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"method": args.name, "output": str(args.output_dir), "test": report["test"],
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)}))


if __name__ == "__main__":
    main()
