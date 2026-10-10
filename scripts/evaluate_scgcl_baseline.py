#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def paired_unit_bootstrap(
    frame: pd.DataFrame,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    grouped = frame.groupby("biological_unit", sort=True).agg(
        n=("n", "sum"),
        raw_error_sum=("raw_error_sum", "sum"),
        method_error_sum=("method_error_sum", "sum"),
    )
    if grouped.empty:
        raise ValueError("no biological units to bootstrap")
    values = grouped.to_numpy(dtype=np.float64)
    observed = float(values[:, 2].sum() / values[:, 0].sum() - values[:, 1].sum() / values[:, 0].sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(values), size=len(values))
        sample = values[selected]
        denominator = sample[:, 0].sum()
        draws[index] = sample[:, 2].sum() / denominator - sample[:, 1].sum() / denominator
    return {
        "difference_method_minus_raw": observed,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_units": int(len(values)),
        "replicates": int(replicates),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corrupted_path = Path(args.corrupted)
    method_path = Path(args.method_output)
    metadata = json.loads((method_path / "metadata.json").read_text())
    prediction = np.load(method_path / "mean.npy", allow_pickle=False)
    selected_genes = np.load(
        method_path / "selected_gene_indices.npy", allow_pickle=False
    ).astype(np.int64)
    adata = ad.read_h5ad(corrupted_path)
    counts = dense(adata.layers["corrupted_counts"]).astype(np.float32)
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    if prediction.shape != counts.shape:
        raise ValueError(f"prediction shape {prediction.shape} != input shape {counts.shape}")
    if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
        raise ValueError("scGCL output cell/gene order does not match the corrupted input")
    if metadata.get("masked_truth_used_for_fit") is not False:
        raise ValueError("scGCL metadata does not certify masked-truth exclusion")

    coordinates = pd.read_parquet(args.coordinates)
    coordinates = coordinates.loc[coordinates["split"].astype(str).eq("test")].copy()
    if coordinates.empty:
        raise ValueError("no locked test coordinates")
    if not (coordinates["original_value"].to_numpy() > 0).all():
        raise ValueError("masked coordinates must have positive original values")
    if not np.allclose(coordinates["corrupted_value"].to_numpy(), 0.0):
        raise ValueError("masked coordinates must be zero in the corrupted matrix")
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    columns = coordinates["gene_index"].to_numpy(dtype=np.int64)
    if not np.allclose(counts[rows, columns], 0.0):
        raise ValueError("coordinate/input corruption mismatch")
    truth = coordinates["original_value"].to_numpy(dtype=np.float64)
    method_values = prediction[rows, columns].astype(np.float64)
    raw_errors = np.log1p(truth)
    method_errors = np.abs(np.log1p(method_values) - np.log1p(truth))
    raw_mae = float(raw_errors.mean())
    method_mae = float(method_errors.mean())

    unit = coordinates[["biological_unit"]].copy()
    unit["n"] = 1
    unit["raw_error_sum"] = raw_errors
    unit["method_error_sum"] = method_errors
    unit_metrics = unit.groupby("biological_unit", as_index=False, sort=True).agg(
        n=("n", "sum"),
        raw_error_sum=("raw_error_sum", "sum"),
        method_error_sum=("method_error_sum", "sum"),
    )
    unit_metrics["raw_masked_log1p_mae"] = unit_metrics["raw_error_sum"] / unit_metrics["n"]
    unit_metrics["scgcl_masked_log1p_mae"] = unit_metrics["method_error_sum"] / unit_metrics["n"]
    comparison = paired_unit_bootstrap(unit_metrics, args.bootstrap_replicates, args.seed)

    splits = pd.read_parquet(args.splits).set_index("cell_id")
    split = splits.loc[cell_ids, "split"].astype(str).to_numpy()
    test_rows = split == "test"
    zero_mask = counts[test_rows] == 0
    zero_fill_fraction = float((prediction[test_rows][zero_mask] > 1e-8).mean())
    selected_mask = np.zeros(counts.shape[1], dtype=bool)
    selected_mask[selected_genes] = True
    selected_coordinate_fraction = float(selected_mask[columns].mean())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    unit_output = output.with_name(output.stem + "_unit_metrics.parquet")
    unit_metrics.to_parquet(unit_output, index=False)
    design = (
        "official-architecture transductive compatibility stability sensitivity; "
        "masked truth excluded; locked test coordinates evaluated once"
        if metadata.get("stability_note")
        else "official transductive standard usage; masked truth excluded; locked test coordinates evaluated once"
    )
    report = {
        "method": metadata["method"],
        "official_commit": metadata["official_commit"],
        "design": design,
        "transductive": bool(metadata["transductive"]),
        "heldout_projection_api": bool(metadata["heldout_projection_api"]),
        "masked_truth_used_for_fit": bool(metadata["masked_truth_used_for_fit"]),
        "epochs": int(metadata["epochs"]),
        "learning_rate": float(metadata["learning_rate"]),
        "upstream_default_learning_rate": float(
            metadata["upstream_default_learning_rate"]
        ),
        "stability_note": metadata.get("stability_note"),
        "selected_genes": int(metadata["selected_genes"]),
        "test_masked_entries": int(len(coordinates)),
        "raw_masked_log1p_mae": raw_mae,
        "scgcl_masked_log1p_mae": method_mae,
        "masked_error_recovery": float(1.0 - method_mae / raw_mae),
        "paired_unit_bootstrap": comparison,
        "test_zero_fill_fraction": zero_fill_fraction,
        "masked_coordinates_in_selected_genes": selected_coordinate_fraction,
        "unit_metrics": str(unit_output),
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
