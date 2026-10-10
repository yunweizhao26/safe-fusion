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

REPOSITORY = Path.cwd()
sys.path.insert(0, str(REPOSITORY / "scripts"))

from selector_attribution import Candidates, fit_scores, selector_features, teacher_feature_names


class BlockFeatures:


    def __init__(self, rows, cols, teachers, gene_mean, gene_dropout, library):
        self.rows, self.cols, self.teachers = rows, cols, teachers
        self.gene_mean, self.gene_dropout, self.library = gene_mean, gene_dropout, library

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index) -> np.ndarray:
        rows, cols = self.rows[index], self.cols[index]
        stack = np.stack([matrix[rows, cols] for matrix in self.teachers])
        return selector_features(stack, self.gene_mean[cols], self.gene_dropout[cols], self.library[rows])


def zero_entries(counts: sparse.csr_matrix, cells: np.ndarray, block_cells: int):

    for start in range(0, len(cells), block_cells):
        block = cells[start:start + block_cells]
        rows, cols = np.nonzero(counts[block].toarray() == 0)
        yield block[rows].astype(np.int32), cols.astype(np.int32)


def is_positive(rows: np.ndarray, cols: np.ndarray, positive_keys: np.ndarray, n_genes: int) -> np.ndarray:
    keys = rows.astype(np.int64) * n_genes + cols
    location = np.minimum(np.searchsorted(positive_keys, keys), max(len(positive_keys) - 1, 0))
    return (positive_keys[location] == keys) if len(positive_keys) else np.zeros(len(keys), dtype=bool)


def gather(matrix: np.ndarray, rows: np.ndarray, cols: np.ndarray, batch_rows: int) -> np.ndarray:
    output = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch_rows):
        output[start:start + batch_rows] = matrix[rows[start:start + batch_rows], cols[start:start + batch_rows]]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--fusion-contract", required=True)
    parser.add_argument("--teacher-contract", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-split", default="development", choices=["development", "validation"])
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument("--block-cells", type=int, default=4096, help="Cells whose zero entries are enumerated together")
    parser.add_argument("--curve-points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.max_fit_rows < 2:
        parser.error("--max-fit-rows must be at least 2")
    timings = {}
    clock = time.perf_counter()

    adata = ad.read_h5ad(args.corrupted)
    counts = sparse.csr_matrix(adata.layers["corrupted_counts"], dtype=np.float32)
    counts.eliminate_zeros()
    n_cells, n_genes = counts.shape
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    fit_cells = np.flatnonzero(split == args.fit_split)
    test_cells = np.flatnonzero(split == "test")
    coordinates = pd.read_parquet(args.coordinates, columns=["cell_index", "gene_index"])
    masked_rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    masked_cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    if counts[masked_rows, masked_cols].any():
        raise ValueError("masked coordinates are not zero in the masked counts")
    positive_keys = np.sort(masked_rows * n_genes + masked_cols)
    teachers = {Path(path).name: np.load(Path(path) / "mean.npy", mmap_mode="r") for path in args.teacher_contract}
    fused = np.load(Path(args.fusion_contract) / "mean.npy", mmap_mode="r")
    for name, matrix in {**teachers, "fused value": fused}.items():
        if matrix.shape != counts.shape:
            raise ValueError(f"{name} has shape {matrix.shape}, the masked counts {counts.shape}")
    feature_names = teacher_feature_names(list(teachers))
    matrices = list(teachers.values())
    timings["load_seconds"] = time.perf_counter() - clock

    clock = time.perf_counter()
    fit_counts = counts[fit_cells]
    n_fit = np.intp(len(fit_cells))
    gene_total = np.asarray(fit_counts.sum(axis=0, dtype=np.float64)).ravel().astype(np.float32)
    gene_mean = np.log1p(np.true_divide(gene_total, n_fit, out=gene_total, casting="unsafe"))
    gene_dropout = (n_fit - np.bincount(fit_counts.indices, minlength=n_genes)).astype(np.float64) / n_fit
    library = np.log1p(np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel().astype(np.float32))
    del fit_counts
    timings["context_seconds"] = time.perf_counter() - clock

    clock = time.perf_counter()
    test_parts = list(zero_entries(counts, test_cells, args.block_cells))
    test_rows = np.concatenate([rows for rows, _ in test_parts])
    test_cols = np.concatenate([cols for _, cols in test_parts])
    del test_parts
    test_labels = is_positive(test_rows, test_cols, positive_keys, n_genes).astype(np.int8)
    timings["test_candidates_seconds"] = time.perf_counter() - clock

    clock = time.perf_counter()
    zeros_per_cell = n_genes - np.diff(counts.indptr)[fit_cells]
    n_fit_candidates = int(zeros_per_cell.sum())
    fit_positive_keys = positive_keys[np.isin(positive_keys // n_genes, fit_cells)]
    if n_fit_candidates <= args.max_fit_rows:
        parts = list(zero_entries(counts, fit_cells, args.block_cells))
        fit_rows = np.concatenate([rows for rows, _ in parts]).astype(np.int64)
        fit_cols = np.concatenate([cols for _, cols in parts]).astype(np.int64)
        sampling = "all candidates"
    else:
        sampled = np.sort(np.random.default_rng(args.seed).choice(n_fit_candidates, size=args.max_fit_rows, replace=False))
        keys, offset = [], 0
        for rows, cols in zero_entries(counts, fit_cells, args.block_cells):
            inside = sampled[(sampled >= offset) & (sampled < offset + len(rows))] - offset
            keys.append(rows[inside].astype(np.int64) * n_genes + cols[inside])
            offset += len(rows)
        joined = np.union1d(np.concatenate(keys), fit_positive_keys)
        fit_rows, fit_cols = joined // n_genes, joined % n_genes
        sampling = "uniform sample of candidates joined with every masked positive"
    fit_labels = is_positive(fit_rows, fit_cols, positive_keys, n_genes).astype(np.int8)
    joined_candidates = int(len(fit_labels))
    if int(fit_labels.sum()) > args.max_fit_rows // 2:
        rng = np.random.default_rng(args.seed)
        half = args.max_fit_rows // 2
        keep = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(fit_labels == 1), size=half, replace=False),
            rng.choice(np.flatnonzero(fit_labels == 0), size=args.max_fit_rows - half, replace=False),
        ]))
        fit_rows, fit_cols, fit_labels = fit_rows[keep], fit_cols[keep], fit_labels[keep]
        sampling += "; half of the fitting rows drawn from each class"
    fit_features = BlockFeatures(fit_rows, fit_cols, matrices, gene_mean, gene_dropout, library)[:]
    timings["fit_candidates_seconds"] = time.perf_counter() - clock

    def candidates(labels: np.ndarray, features) -> Candidates:
        zeros = np.zeros(len(labels), dtype=np.float32)
        return Candidates(rows=np.arange(len(labels), dtype=np.int64), cols=np.zeros(len(labels), dtype=np.int64),
                          labels=labels, features=features, fused_values=zeros, teacher_values=zeros,
                          truth_values=zeros, feature_names=feature_names)

    clock = time.perf_counter()
    _, test_score, model_report = fit_scores(
        "full", "mlp", candidates(fit_labels, fit_features),
        candidates(test_labels, BlockFeatures(test_rows, test_cols, matrices, gene_mean, gene_dropout, library)),
        args.max_fit_rows, args.score_batch_rows, args.seed,
    )
    timings["fit_and_score_seconds"] = time.perf_counter() - clock

    clock = time.perf_counter()
    test_fused_value = gather(fused, test_rows, test_cols, args.score_batch_rows)
    ranked_labels = test_labels[np.argsort(-test_score, kind="stable")]
    cumulative = np.cumsum(ranked_labels, dtype=np.int64)
    positives = int(test_labels.sum())
    curve = []
    for requested in np.linspace(0.001, 1.0, args.curve_points):
        selected = max(1, int(round(float(requested) * len(test_labels))))
        true_positive = int(cumulative[selected - 1])
        precision, recall = true_positive / selected, true_positive / positives
        curve.append({
            "requested_fill_fraction": float(requested), "realized_fill_fraction": selected / len(test_labels),
            "n_selected": selected, "n_true_positive": true_positive, "masked_precision": precision,
            "masked_recall": recall,
            "masked_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        })
    metrics = {"roc_auc": float(roc_auc_score(test_labels, test_score)),
               "pr_auc": float(average_precision_score(test_labels, test_score))}
    timings["metrics_seconds"] = time.perf_counter() - clock

    clock = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "test_candidates.npz", rows=test_rows, cols=test_cols, labels=test_labels)
    np.save(output / "test_score.npy", test_score.astype(np.float32))
    np.save(output / "test_fused_value.npy", test_fused_value)
    timings["write_seconds"] = time.perf_counter() - clock
    report = {
        "architecture": "mlp",
        "max_fit_rows": args.max_fit_rows,
        "feature_names": list(feature_names),
        "cells": int(n_cells), "genes": int(n_genes),
        "fit_split": args.fit_split, "fit_cells": int(len(fit_cells)), "test_cells": int(len(test_cells)),
        "fit_candidates": n_fit_candidates, "fit_masked_positives": int(len(fit_positive_keys)),
        "fit_sampling": sampling, "fit_joined_candidates": joined_candidates,
        "fit_sample_candidates": int(len(fit_labels)), "fit_sample_positives": int(fit_labels.sum()),
        "test": {"n_zeros": int(len(test_labels)), "n_masked_positives": positives, **metrics},
        "models": {"mlp": model_report},
        "exact_budget_curve": curve,
        "timings": timings,
        "test_labels_used_for_fit": False,
    }
    (output / "selector_report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({key: report[key] for key in ("fit_candidates", "fit_masked_positives", "fit_sample_candidates", "test", "timings")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
