#!/usr/bin/env python3
"""Compute exact-fill masked-positive F1 curves for the paper methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "paper_evidence"
PANCREAS = ROOT / "artifacts" / "pancreas_runs" / "0b2469810675-45c81b160d78"
COLON = ROOT / "artifacts" / "colon_runs" / "0b2469810675-c0db6f963e94"


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def validate_contract(contract: Path, adata: ad.AnnData) -> Path:
    metadata = json.loads((contract / "metadata.json").read_text())
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    if metadata["cell_ids"] != cell_ids:
        raise ValueError(f"Cell order differs for {contract}")
    if metadata["gene_ids"] != gene_ids:
        raise ValueError(f"Gene order differs for {contract}")
    return contract / "mean.npy"


def ranked_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    budgets: np.ndarray,
    seed: int,
) -> list[dict[str, float]]:
    # Random tie breaking is independent of the labels and prevents the stored
    # cell or gene order from deciding large ALRA zero ties.
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(scores)), -np.nan_to_num(scores, nan=-np.inf)))
    ranked_labels = labels[order]
    cumulative_true = np.cumsum(ranked_labels, dtype=np.int64)
    positives = int(labels.sum())
    zeros = len(labels)
    curve = []
    for budget in budgets:
        selected = max(1, int(round(float(budget) * zeros)))
        true_positive = int(cumulative_true[selected - 1])
        curve.append(
            {
                "requested_fill_fraction": float(budget),
                "n_selected": selected,
                "n_true_positive": true_positive,
                "n_masked_positives": positives,
                "n_zeros": zeros,
            }
        )
    return curve


def unit_curves(
    corrupted_path: Path,
    coordinates_path: Path,
    splits_path: Path,
    contracts: dict[str, Path],
    budgets: np.ndarray,
    seed: int,
) -> dict[str, list[dict[str, float]]]:
    adata = ad.read_h5ad(corrupted_path)
    counts = dense(adata.layers["corrupted_counts"])
    split = (
        pd.read_parquet(splits_path)
        .set_index("cell_id")
        .loc[adata.obs_names.astype(str), "split"]
        .to_numpy()
    )
    test_cells = np.flatnonzero(split == "test")
    local_rows, cols = np.where(counts[test_cells] == 0)
    rows = test_cells[local_rows]

    masked = np.zeros(counts.shape, dtype=bool)
    coordinates = pd.read_parquet(coordinates_path)
    masked[
        coordinates["cell_index"].to_numpy(dtype=int),
        coordinates["gene_index"].to_numpy(dtype=int),
    ] = True
    labels = masked[rows, cols].astype(np.int8)

    curves = {}
    for method, contract in contracts.items():
        mean_path = validate_contract(contract, adata)
        mean = np.load(mean_path, mmap_mode="r", allow_pickle=False)
        scores = np.asarray(mean[rows, cols], dtype=np.float32)
        curves[method] = ranked_curve(scores, labels, budgets, seed)
    return curves


def safe_fusion_curves(dataset: str) -> list[list[dict[str, float]]]:
    if dataset == "Pancreas":
        paths = [
            EVIDENCE
            / "pancreas_crossfit"
            / f"fold_{fold}"
            / "selector_exact_budget"
            / "logistic"
            / "calibration_report.json"
            for fold in range(3)
        ]
    elif dataset == "Colon":
        paths = [EVIDENCE / "selector_exact_budget" / "colon" / "logistic" / "calibration_report.json"]
    elif dataset == "CRISPRa":
        paths = [EVIDENCE / "selector_exact_budget" / "norman_crispra" / "logistic" / "calibration_report.json"]
    else:
        raise ValueError(dataset)

    curves = []
    for path in paths:
        report = json.loads(path.read_text())
        points = []
        for point in report["exact_budget_curve"]:
            points.append(
                {
                    "requested_fill_fraction": point["requested_fill_fraction"],
                    "n_selected": point["n_selected"],
                    "n_true_positive": point["n_true_positive"],
                    "n_masked_positives": report["test"]["n_masked_positives"],
                    "n_zeros": report["test"]["n_zeros"],
                }
            )
        curves.append(points)
    return curves


def pool_curves(curves: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    point_count = len(curves[0])
    if any(len(curve) != point_count for curve in curves):
        raise ValueError("Curve lengths differ")
    pooled = []
    for index in range(point_count):
        points = [curve[index] for curve in curves]
        requested = float(points[0]["requested_fill_fraction"])
        if any(abs(float(point["requested_fill_fraction"]) - requested) > 1e-12 for point in points):
            raise ValueError("Curve budgets differ")
        selected = int(sum(point["n_selected"] for point in points))
        true_positive = float(sum(point["n_true_positive"] for point in points))
        positives = int(sum(point["n_masked_positives"] for point in points))
        zeros = int(sum(point["n_zeros"] for point in points))
        precision = true_positive / selected
        recall = true_positive / positives
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        pooled.append(
            {
                "requested_fill_fraction": requested,
                "realized_fill_fraction": selected / zeros,
                "n_selected": selected,
                "n_true_positive": true_positive,
                "n_masked_positives": positives,
                "n_zeros": zeros,
                "masked_precision": precision,
                "masked_recall": recall,
                "masked_f1": f1,
            }
        )
    return pooled


def random_expectation(reference: list[dict[str, float]]) -> list[dict[str, float]]:
    curve = []
    for point in reference:
        expected_true = point["n_selected"] * point["n_masked_positives"] / point["n_zeros"]
        precision = point["n_masked_positives"] / point["n_zeros"]
        recall = expected_true / point["n_masked_positives"]
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        curve.append(
            {
                **point,
                "n_true_positive": expected_true,
                "masked_precision": precision,
                "masked_recall": recall,
                "masked_f1": f1,
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "selector_f1_fillrate_baselines_1000_points.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=EVIDENCE / "selector_f1_fillrate_baselines_summary.json",
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    budgets = np.linspace(0.001, 1.0, 1000)
    pancreas_corrupted = PANCREAS / "data" / "pancreas_islets" / "corrupted" / "mask_010.h5ad"
    pancreas_coordinates = PANCREAS / "data" / "pancreas_islets" / "coordinates" / "mask_010.parquet"
    pancreas_units = []
    for fold in range(3):
        fold_root = EVIDENCE / "pancreas_crossfit" / f"fold_{fold}"
        pancreas_units.append(
            unit_curves(
                pancreas_corrupted,
                pancreas_coordinates,
                fold_root / "splits.parquet",
                {
                    "Weighted kNN": fold_root / "graph_smooth",
                    "SVD": fold_root / "svd_impute",
                    "ALRA": fold_root / "alra",
                    "SAVER": EVIDENCE / "baselines" / "saver" / "pancreas_mask_010",
                    "MAGIC": EVIDENCE / "baselines" / "magic" / "pancreas",
                    "scVI": EVIDENCE / "baselines" / "scvi" / "pancreas",
                    "scGCL": EVIDENCE / "baselines" / "scgcl" / "pancreas",
                    "scGPT": EVIDENCE / "baselines" / "scgpt_mvc" / "pancreas",
                },
                budgets,
                args.seed + 100 * fold,
            )
        )

    colon_units = [
        unit_curves(
            COLON / "data" / "colon_epithelial" / "corrupted" / "mask_010.h5ad",
            COLON / "data" / "colon_epithelial" / "coordinates" / "mask_010.parquet",
            COLON / "data" / "colon_epithelial" / "splits.parquet",
            {
                "Weighted kNN": COLON / "methods" / "standardized" / "colon_epithelial" / "mask_010" / "graph_smooth",
                "SVD": COLON / "methods" / "standardized" / "colon_epithelial" / "mask_010" / "svd_impute",
                "ALRA": EVIDENCE / "baselines" / "alra" / "colon_mask_010",
                "SAVER": EVIDENCE / "baselines" / "saver" / "colon_mask_010",
                "MAGIC": EVIDENCE / "baselines" / "magic" / "colon",
                "scVI": EVIDENCE / "baselines" / "scvi" / "colon",
                "scGCL": EVIDENCE / "baselines" / "scgcl" / "colon",
                "scGPT": EVIDENCE / "baselines" / "scgpt_mvc" / "colon",
            },
            budgets,
            args.seed + 1000,
        )
    ]

    norman_root = EVIDENCE / "norman_crispra"
    crispra_units = [
        unit_curves(
            norman_root / "corrupted.h5ad",
            norman_root / "coordinates.parquet",
            norman_root / "splits.parquet",
            {
                "Weighted kNN": norman_root / "methods" / "graph_smooth",
                "SVD": norman_root / "methods" / "svd_impute",
                "ALRA": EVIDENCE / "baselines" / "alra" / "norman_mask_010",
                "SAVER": EVIDENCE / "baselines" / "saver" / "norman_full_mask_010",
                "MAGIC": EVIDENCE / "baselines" / "magic" / "norman",
                "scVI": EVIDENCE / "baselines" / "scvi" / "norman",
                "scGCL": EVIDENCE / "baselines" / "scgcl" / "norman",
                "scGPT": EVIDENCE / "baselines" / "scgpt_mvc" / "norman",
            },
            budgets,
            args.seed + 2000,
        )
    ]

    dataset_units = {
        "Pancreas": pancreas_units,
        "Colon": colon_units,
        "CRISPRa": crispra_units,
    }
    rows = []
    summary = {}
    for dataset, units in dataset_units.items():
        methods: dict[str, list[dict[str, float]]] = {
            "Safe Fusion": pool_curves(safe_fusion_curves(dataset))
        }
        for method in (
            "Weighted kNN",
            "SVD",
            "ALRA",
            "SAVER",
            "MAGIC",
            "scVI",
            "scGCL",
            "scGPT",
        ):
            methods[method] = pool_curves([unit[method] for unit in units])

        summary[dataset] = {}
        for method, curve in methods.items():
            for point in curve:
                rows.append({"dataset": dataset, "method": method, **point})
            peak = max(curve, key=lambda point: point["masked_f1"])
            summary[dataset][method] = {
                "peak_fill_percent": 100.0 * peak["realized_fill_fraction"],
                "peak_f1": peak["masked_f1"],
                "f1_at_2_percent": curve[19]["masked_f1"],
                "f1_at_5_percent": curve[49]["masked_f1"],
                "f1_at_10_percent": curve[99]["masked_f1"],
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
