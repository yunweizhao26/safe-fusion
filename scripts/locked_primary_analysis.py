#!/usr/bin/env python3
"""Validation-select a baseline and bootstrap the locked primary endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def prediction(contract: Path) -> np.ndarray:
    if (contract / "failure.json").exists():
        reason = json.loads((contract / "failure.json").read_text()).get("reason", "method failure")
        raise RuntimeError(reason)
    return np.load(contract / "mean.npy", mmap_mode="r")


def per_unit_mae(coordinates: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    row = coordinates["cell_index"].to_numpy(dtype=int)
    col = coordinates["gene_index"].to_numpy(dtype=int)
    error = np.abs(np.log1p(values[row, col]) - np.log1p(coordinates["original_value"].to_numpy(dtype=float)))
    frame = pd.DataFrame({"biological_unit": coordinates["biological_unit"].astype(str), "absolute_error": error})
    return frame.groupby("biological_unit", as_index=False).agg(log1p_mae=("absolute_error", "mean"), n=("absolute_error", "size"))


def bootstrap_relative(sf: pd.DataFrame, baseline: pd.DataFrame, replicates: int, seed: int) -> dict:
    paired = sf.merge(baseline, on="biological_unit", suffixes=("_sf", "_baseline"), validate="one_to_one")
    sf_mean = float(np.average(paired["log1p_mae_sf"], weights=paired["n_sf"]))
    baseline_mean = float(np.average(paired["log1p_mae_baseline"], weights=paired["n_baseline"]))
    observed = 1.0 - sf_mean / baseline_mean
    payload = {
        "n_test_biological_units": len(paired), "safe_fusion_log1p_mae": sf_mean,
        "baseline_log1p_mae": baseline_mean, "relative_improvement": observed,
        "absolute_improvement": baseline_mean - sf_mean,
    }
    if len(paired) < 2:
        payload.update({"relative_improvement_ci95": None, "inference": "descriptive_only_single_test_unit"})
        return payload
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates)
    for i in range(replicates):
        sampled = rng.integers(0, len(paired), size=len(paired))
        current = paired.iloc[sampled]
        current_sf = np.average(current["log1p_mae_sf"], weights=current["n_sf"])
        current_base = np.average(current["log1p_mae_baseline"], weights=current["n_baseline"])
        boot[i] = 1.0 - current_sf / current_base
    payload.update({
        "relative_improvement_ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": replicates,
        "probability_improvement_gt_0": float(np.mean(boot > 0)),
        "probability_improvement_ge_5pct": float(np.mean(boot >= 0.05)),
        "inference": "paired_cluster_bootstrap_over_biological_units",
    })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--contracts-root", required=True)
    parser.add_argument("--baselines", nargs="+", required=True)
    parser.add_argument("--safe-fusion", default="safe_fusion")
    parser.add_argument("--safe-fusion-contract", default=None)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    coordinates = pd.read_parquet(args.coordinates)
    validation = coordinates[coordinates["split"].eq("validation")].copy()
    test = coordinates[coordinates["split"].eq("test")].copy()
    if validation.empty or test.empty:
        raise ValueError("both validation and locked test coordinates are required")
    root = Path(args.contracts_root)
    selection = []
    per_method: dict[str, dict[str, pd.DataFrame]] = {}
    for method in [*args.baselines, args.safe_fusion]:
        try:
            contract = Path(args.safe_fusion_contract) if method == args.safe_fusion and args.safe_fusion_contract else root / method
            values = prediction(contract)
            validation_units = per_unit_mae(validation, values)
            test_units = per_unit_mae(test, values)
            pooled_validation = float(np.average(validation_units["log1p_mae"], weights=validation_units["n"]))
            selection.append({"method": method, "validation_log1p_mae": pooled_validation, "status": "ok"})
            per_method[method] = {"validation": validation_units, "test": test_units}
        except Exception as exc:
            selection.append({"method": method, "validation_log1p_mae": None, "status": str(exc)})
    eligible = [row for row in selection if row["method"] in args.baselines and row["status"] == "ok"]
    if not eligible:
        raise RuntimeError("no successful baseline was available for validation selection")
    chosen = min(eligible, key=lambda row: row["validation_log1p_mae"])["method"]
    comparison = bootstrap_relative(per_method[args.safe_fusion]["test"], per_method[chosen]["test"], args.replicates, args.seed)
    payload = {
        "dataset": args.dataset, "corruption": args.corruption,
        "selection_policy": "minimum pooled validation log1p MAE; locked test excluded",
        "validation_selection": selection, "selected_baseline": chosen,
        "locked_test_comparison": comparison,
        "per_unit": {
            args.safe_fusion: per_method[args.safe_fusion]["test"].to_dict(orient="records"),
            chosen: per_method[chosen]["test"].to_dict(orient="records"),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
