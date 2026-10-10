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
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import UNIT_FRACTIONS
from selector_attribution import Candidates, fit_scores, selector_features, teacher_feature_names
from stacked_selector_baselines import candidate_keys

TEACHERS = ("gene_median", "svd_impute", "graph_smooth", "magic_inductive", "scvi_inductive")
FEATURE_SETS = ("base", "detection")

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def teacher_values(contract: Path, cell_ids: list[str], gene_ids: list[str], rows: np.ndarray,
                   cols: np.ndarray) -> np.ndarray:
    metadata = json.loads((contract / "metadata.json").read_text())
    if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
        raise ValueError(f"cell or gene order of {contract} differs from the input")
    if metadata["scale"] != "counts":
        raise ValueError(f"{contract} is not on the count scale")
    mean = np.load(contract / "mean.npy", mmap_mode="r", allow_pickle=False)
    return np.asarray(mean[rows, cols], dtype=np.float64).astype(np.float32)

def negative_binomial_detection(mu: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return -np.expm1(-theta * np.log1p(np.maximum(mu, 0.0) / theta))

def detection_features(values: np.ndarray, theta: np.ndarray) -> np.ndarray:

    mu = np.maximum(values.astype(np.float64), 0.0)
    columns = [-np.expm1(-mu[index]) for index in range(len(values) - 1)]
    columns.append(negative_binomial_detection(mu[-1], theta))
    return np.column_stack(columns).astype(np.float32)

def mean_f1(scores: np.ndarray, labels: np.ndarray) -> dict:

    cumulative = np.cumsum(labels[np.argsort(-scores, kind="stable")], dtype=np.int64)
    selected = [max(1, int(round(fraction * len(labels)))) for fraction in UNIT_FRACTIONS]
    true_positive = [int(cumulative[k - 1]) for k in selected]
    positives = int(labels.sum())
    f1 = [2.0 * tp / (k + positives) for k, tp in zip(selected, true_positive)]
    return {"mean_f1_1_to_10": float(np.mean(f1)), "f1": f1, "n_selected": selected, "n_true_positive": true_positive,
            "average_precision": float(average_precision_score(labels, scores)),
            "n_candidates": int(len(labels)), "n_positives": positives}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--teachers-root", type=Path, required=True,
                        help="Directory with the five teacher contracts " + ", ".join(TEACHERS) + ".")
    parser.add_argument("--teacher-names", nargs=5, default=list(TEACHERS),
                        help="Contract directory names under --teachers-root, in the production order.")
    parser.add_argument("--theta-root", type=Path, default=None,
                        help="Directory with theta.npy and model_index.npy of the scVI teacher (default: its contract).")
    parser.add_argument("--fit-split", choices=["development", "validation"], required=True)
    parser.add_argument("--features", choices=FEATURE_SETS, required=True)
    parser.add_argument("--evaluation-column", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    started = time.perf_counter()

    adata = ad.read_h5ad(args.input)
    counts = dense(adata.layers["corrupted_counts"]).astype(np.float32)
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    split_frame = pd.read_parquet(args.splits).set_index("cell_id").loc[cell_ids]
    split = split_frame["split"].astype(str).to_numpy()
    coordinates = pd.read_parquet(args.coordinates)
    positive = np.zeros(counts.shape, dtype=bool)
    positive[coordinates["cell_index"].to_numpy(dtype=np.int64), coordinates["gene_index"].to_numpy(dtype=np.int64)] = True
    if np.any(counts[positive] != 0):
        raise ValueError("positives must be zeros of the input")

    fit_mask = split == args.fit_split
    test_mask = split == "test"
    gene_mean = np.log1p(np.mean(counts[fit_mask], axis=0))
    gene_dropout = np.mean(counts[fit_mask] <= 0, axis=0)
    log_library = np.log1p(counts.sum(axis=1))

    fit_rows, fit_cols = candidate_keys(counts, fit_mask, positive, args.max_fit_rows, args.seed)
    test_rows, test_cols = np.where((counts == 0) & test_mask[:, None])
    fit_labels = positive[fit_rows, fit_cols].astype(np.int8)
    test_labels = positive[test_rows, test_cols].astype(np.int8)

    contracts = [args.teachers_root / name for name in args.teacher_names]
    theta_root = args.theta_root or contracts[-1]
    if args.features == "detection" or args.evaluation_column:
        theta = np.load(theta_root / "theta.npy")
        model_index = np.load(theta_root / "model_index.npy")
        if len(model_index) != len(cell_ids) or theta.shape[1] != len(gene_ids):
            raise ValueError(f"{theta_root} does not match the input")
    names = teacher_feature_names(list(args.teacher_names))
    if args.features == "detection":
        names = names + tuple(f"detection:{name}" for name in args.teacher_names)

    def candidates(rows: np.ndarray, cols: np.ndarray, labels: np.ndarray) -> tuple[Candidates, np.ndarray]:
        values = np.stack([teacher_values(contract, cell_ids, gene_ids, rows, cols) for contract in contracts])
        features = selector_features(values, gene_mean[cols], gene_dropout[cols], log_library[rows])
        if args.features == "detection":
            features = np.column_stack([features, detection_features(values, theta[model_index[rows], cols])])
        placeholder = np.zeros(len(rows), dtype=np.float32)
        return Candidates(rows=rows, cols=cols, labels=labels, features=features, fused_values=placeholder,
                          teacher_values=values, truth_values=placeholder, feature_names=names), values

    fit_candidates, _ = candidates(fit_rows, fit_cols, fit_labels)
    test_candidates, test_values = candidates(test_rows, test_cols, test_labels)
    fit_score, test_score, model_report = fit_scores(
        "full", "mlp", fit_candidates, test_candidates, args.max_fit_rows, args.score_batch_rows, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "test_scores.npy", test_score.astype(np.float32), allow_pickle=False)
    report = {
        "input": str(args.input), "coordinates": str(args.coordinates), "splits": str(args.splits),
        "teachers": [str(path) for path in contracts], "theta_root": str(theta_root),
        "features": args.features, "feature_names": list(names), "fit_split": args.fit_split,
        "fit": {"n_zeros": int(len(fit_labels)), "n_positives": int(fit_labels.sum()),
                "roc_auc": float(roc_auc_score(fit_labels, fit_score)),
                "average_precision": float(average_precision_score(fit_labels, fit_score))},
        "test_candidate_order": "row-major (cell, gene) order of the zeros of the held-out cells",
        "test_labels_used_for_training": False,
        "seed": args.seed,
        "model": model_report,
    }
    if test_labels.any():
        report["test"] = {"n_zeros": int(len(test_labels)), "n_positives": int(test_labels.sum()),
                          "average_precision": float(average_precision_score(test_labels, test_score))}
    if args.evaluation_column:
        groups = split_frame[args.evaluation_column].astype(str).to_numpy()[test_rows]
        scvi_theta = theta[model_index[test_rows], test_cols]
        references = {
            "scVI teacher": test_values[-1].astype(np.float64),
            "scVI teacher P(X>0)": negative_binomial_detection(test_values[-1].astype(np.float64), scvi_theta),
        }
        report["evaluation"] = {}
        for group in sorted(set(groups)):
            keep = groups == group
            labels = test_labels[keep]
            report["evaluation"][group] = {"selector": mean_f1(test_score[keep], labels)}
            for name, score in references.items():
                order = np.lexsort((np.random.default_rng(args.seed).random(int(keep.sum())), -score[keep]))
                tie_broken = np.empty(len(order))
                tie_broken[order] = -np.arange(len(order), dtype=np.float64)
                report["evaluation"][group][name] = mean_f1(tie_broken, labels)
    report["elapsed_seconds"] = time.perf_counter() - started
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"output": str(args.output_dir), "features": args.features,
                      "evaluation": {group: {name: round(100 * values["mean_f1_1_to_10"], 3)
                                             for name, values in by.items()}
                                     for group, by in report.get("evaluation", {}).items()},
                      "elapsed_seconds": round(report["elapsed_seconds"], 1)}), flush=True)

if __name__ == "__main__":
    main()
