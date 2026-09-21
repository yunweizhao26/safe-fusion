#!/usr/bin/env python3
"""Apply the frozen q=0.90, tau=0.081 Safe Fusion decision rule."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.contracts import write_output_contract  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--fusion-contract", required=True)
    parser.add_argument("--teacher-contract", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence-quantile", type=float, default=0.90)
    parser.add_argument("--fill-budget", type=float, default=0.081)
    parser.add_argument("--fallback", choices=["teacher_max", "raw"], default="teacher_max")
    parser.add_argument("--method-name", default="safe_fusion_sparse")
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        adata.obs_names.astype(str), "split"
    ].to_numpy()
    calibration = np.isin(split, ["development", "validation"])
    calibration_splits = sorted({str(value) for value in np.unique(split[calibration])})
    zero = counts == 0
    calibration_zero = zero & calibration[:, None]

    source = Path(args.fusion_contract)
    fused = np.load(source / "mean.npy", allow_pickle=False)
    variance = np.load(source / "variance.npy", allow_pickle=False)
    confidence = -np.log(np.clip(variance, 1e-8, None))
    confidence_threshold = float(np.quantile(confidence[calibration_zero], args.confidence_quantile))
    use_fusion = confidence >= confidence_threshold

    teachers = [np.load(Path(path) / "mean.npy", allow_pickle=False) for path in args.teacher_contract]
    fallback = np.max(np.stack(teachers), axis=0)
    safe_score = np.where(use_fusion, fused, fallback if args.fallback == "teacher_max" else 0.0)
    score_threshold = float(np.quantile(safe_score[calibration_zero], 1.0 - args.fill_budget))
    fill = zero & (safe_score >= score_threshold)
    if args.fallback == "raw":
        fill &= use_fusion
    output = counts.copy()
    output[fill] = safe_score[fill]

    metadata = json.loads((source / "metadata.json").read_text())
    metadata["method"] = args.method_name
    metadata["parameters"] = {
        **metadata.get("parameters", {}),
        "decision_layer": {
            "confidence_quantile": args.confidence_quantile,
            "fill_budget": args.fill_budget,
            "thresholds_fit_on": calibration_splits,
            "confidence_threshold": confidence_threshold,
            "score_threshold": score_threshold,
            "test_fill_fraction": float(fill[split == "test"].sum() / max(1, zero[split == "test"].sum())),
            "test_values_used_for_thresholds": False,
            "fallback": args.fallback,
            "teacher_contracts": args.teacher_contract,
        },
    }
    write_output_contract(
        args.output,
        output,
        metadata,
        variance=variance,
        fill_score=safe_score.astype(np.float32),
    )
    print(json.dumps(metadata["parameters"]["decision_layer"], indent=2))


if __name__ == "__main__":
    main()
