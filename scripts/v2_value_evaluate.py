#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import fraction_name, load_units_manifest
from v2_value_select import bootstrap_weights, interval

EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"
VALUE_ROOT = EVIDENCE / "review_round3" / "value_v2_ablations" / "value"
TRANSFER = EVIDENCE / "review_round2" / "thinning_transfer" / "mask_trained"
MANIFEST = EVIDENCE / "review_round2" / "leakage_free" / "units_manifest.json"
COLON_CROSSFIT = EVIDENCE / "review_round3" / "colon_crossfit"
COLON_MANIFEST = COLON_CROSSFIT / "units_manifest.json"
CURRENT = "conditional"
FRACTIONS = (0.01, 0.05, 0.10)
THINNING_FILL = 0.05
TRUE_COUNT_BINS = [0, 1, 3, 10, np.inf]
TRUE_COUNT_LABELS = ["1", "2-3", "4-10", ">10"]

THINNING_UNITS = {
    **{f"colon_thinning_050_{fold}": ("Colon, 50%", 0.5, "donor", COLON_CROSSFIT / "thinning" / "mask_trained") for fold in range(3)},
    **{f"colon_thinning_025_{fold}": ("Colon, 25%", 0.25, "donor", COLON_CROSSFIT / "thinning" / "mask_trained") for fold in range(3)},
    **{f"pancreas_thinning_050_{fold}": ("Pancreas, 50%", 0.5, "donor", TRANSFER) for fold in range(3)},
    "norman_thinning_050": ("Norman, 50%", 0.5, "target", TRANSFER),
}

def values_at(contract_root: Path, values: list[str], rows: np.ndarray, cols: np.ndarray, production: Path) -> dict[str, np.ndarray]:

    result = {}
    for value in values:
        path = production if value == CURRENT else contract_root / value
        result[value] = np.asarray(np.load(path / "mean.npy", mmap_mode="r")[rows, cols], dtype=np.float64)
    return result

def filled(selector_dir: Path, fraction: float, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    path = selector_dir / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}" / "mean.npy"
    return np.asarray(np.load(path, mmap_mode="r")[rows, cols]) > 0

def masked_unit_frame(unit, values: list[str], contract_root: Path) -> pd.DataFrame:
    adata = ad.read_h5ad(unit.corrupted)
    cell_ids = adata.obs_names.astype(str)
    groups = adata.obs[unit.unit_column].astype(str).to_numpy()
    counts = adata.layers["corrupted_counts"]
    split = pd.read_parquet(unit.splits).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    coordinates = pd.read_parquet(unit.coordinates)
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    keep = split[rows] == "test"
    rows, cols = rows[keep], cols[keep]
    y = coordinates["original_value"].to_numpy(dtype=np.float64)[keep]
    if np.any(np.asarray(counts[rows, cols]).ravel() != 0) or np.any(y <= 0):
        raise ValueError(f"{unit.key}: held-out masked positives are not zeros with a positive count")
    production = unit.contracts["Gene median"].parent / "safe_fusion"
    frame = pd.DataFrame({"unit": unit.key, "group": groups[rows], "count": y,
                          "true_count": pd.cut(np.round(y), TRUE_COUNT_BINS, labels=TRUE_COUNT_LABELS).astype(str)})
    for value, array in values_at(contract_root / unit.key, values, rows, cols, production).items():
        frame[f"value:{value}"] = array
    for fraction in FRACTIONS:
        frame[f"selected:{fraction}"] = filled(unit.selector_dir, fraction, rows, cols)
    return frame

def thinning_unit_frame(key: str, values: list[str], contract_root: Path) -> pd.DataFrame:
    dataset, retained, unit_column, transfer_root = THINNING_UNITS[key]
    root = transfer_root / key
    adata = ad.read_h5ad(root / "input" / "hybrid.h5ad")
    cell_ids = adata.obs_names.astype(str)
    groups = adata.obs[unit_column].astype(str).to_numpy()
    split = pd.read_parquet(root / "input" / "splits.parquet").set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    coordinates = pd.read_parquet(root / "input" / "coordinates.parquet")
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    keep = split[rows] == "test"
    rows, cols = rows[keep], cols[keep]
    x = coordinates["original_value"].to_numpy(dtype=np.float64)[keep]
    fold = key.rpartition("_")[2] if key.startswith(("colon", "pancreas")) else None
    group = groups[rows] if fold is None else np.char.add(f"fold_{fold}:", groups[rows].astype(str))
    frame = pd.DataFrame({"unit": key, "dataset": dataset, "group": group, "count": x, "expected_count": retained * x,
                          "selected": filled(root / "selector", THINNING_FILL, rows, cols)})
    for value, array in values_at(contract_root / key, values, rows, cols, root / "safe_fusion").items():
        frame[f"value:{value}"] = array
    return frame

def summarize(frame: pd.DataFrame, statistics: dict, values: list[str], draws: int, seed: int, labels: dict) -> list[dict]:

    rows = []
    grouped_index = frame.groupby("group", sort=True).indices
    group_order = sorted(grouped_index)
    weights = bootstrap_weights(len(group_order), draws, seed)
    for name, function in statistics.items():
        samples = {}
        for value in values:
            numerator, denominator = function(frame, value)
            num = np.array([numerator[grouped_index[g]].sum() for g in group_order])
            den = np.array([denominator[grouped_index[g]].sum() for g in group_order])
            samples[value] = (weights @ num) / (weights @ den)
            low, high = interval(samples[value][1:])
            rows.append({**labels, "statistic": name, "value": value, "estimate": float(samples[value][0]),
                         "lower": low, "upper": high, "groups": len(group_order), "entries": int((denominator > 0).sum())})
        for value in values:
            if value == CURRENT:
                continue
            difference = samples[value] - samples[CURRENT]
            low, high = interval(difference[1:])
            rows.append({**labels, "statistic": name, "value": f"{value} minus {CURRENT}", "estimate": float(difference[0]),
                         "lower": low, "upper": high, "groups": len(group_order), "entries": None})
    return rows

def log_value(frame: pd.DataFrame, value: str) -> np.ndarray:
    return np.log1p(np.maximum(frame[f"value:{value}"].to_numpy(), 0.0))

def masked_statistics() -> dict:
    statistics = {"log_error": lambda f, v: (np.abs(log_value(f, v) - np.log1p(f["count"].to_numpy())), np.ones(len(f)))}
    for label in TRUE_COUNT_LABELS:
        statistics[f"log_error_true_count_{label}"] = (
            lambda f, v, label=label: (np.abs(log_value(f, v) - np.log1p(f["count"].to_numpy())) * (f["true_count"] == label).to_numpy(),
                                       (f["true_count"] == label).to_numpy().astype(float)))
    for fraction in FRACTIONS:
        def removed(f, v, fraction=fraction):
            truth = np.log1p(f["count"].to_numpy())
            error = np.where(f[f"selected:{fraction}"].to_numpy(), np.abs(log_value(f, v) - truth), truth)
            return 100.0 * (truth - error), truth
        statistics[f"error_removed_{int(round(100 * fraction))}pct"] = removed
    return statistics

def thinning_statistics() -> dict:
    statistics = {}
    for scope in ("all", "filled_5pct"):
        def mask(f, scope=scope):
            return np.ones(len(f)) if scope == "all" else f["selected"].to_numpy().astype(float)
        statistics[f"bias_{scope}"] = lambda f, v, mask=mask: (
            (log_value(f, v) - np.log1p(f["expected_count"].to_numpy())) * mask(f), mask(f))
        statistics[f"error_vs_x_{scope}"] = lambda f, v, mask=mask: (
            np.abs(log_value(f, v) - np.log1p(f["count"].to_numpy())) * mask(f), mask(f))
        statistics[f"inserted_log1p_{scope}"] = lambda f, v, mask=mask: (log_value(f, v) * mask(f), mask(f))
    return statistics

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--colon-manifest", type=Path, default=COLON_MANIFEST,
                        help="Colon units that replace the colon units of --units-manifest.")
    parser.add_argument("--contract-root", type=Path, default=VALUE_ROOT / "contracts")
    parser.add_argument("--selection", type=Path, default=VALUE_ROOT / "selection" / "selection.json")
    parser.add_argument("--values", nargs="+", default=None, help="Values to evaluate (default: current and chosen).")
    parser.add_argument("--output-dir", type=Path, default=VALUE_ROOT / "test")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--pancreas-thinning-root", type=Path, default=None,
        help="Overrides the mask-trained root of pancreas_thinning_050_* units "
        "(default: the production-fold review_round2/thinning_transfer root).",
    )
    args = parser.parse_args()
    if args.pancreas_thinning_root is not None:
        for fold in range(3):
            THINNING_UNITS[f"pancreas_thinning_050_{fold}"] = ("Pancreas, 50%", 0.5, "donor", args.pancreas_thinning_root)
    chosen = json.loads(args.selection.read_text())["chosen"] if args.values is None else None
    values = args.values or list(dict.fromkeys([CURRENT, chosen]))

    rows = []
    units = [unit for unit in load_units_manifest(args.units_manifest) if unit.dataset != "Colon"]
    units += load_units_manifest(args.colon_manifest)
    for dataset in ("Pancreas", "Colon", "CRISPRa"):
        frame = pd.concat([masked_unit_frame(unit, values, args.contract_root) for unit in units if unit.dataset == dataset],
                          ignore_index=True)
        rows += summarize(frame, masked_statistics(), values, args.draws, args.seed, {"benchmark": "masked", "dataset": dataset})
        print(f"masked {dataset}: {len(frame)} positives", flush=True)
    thinning = pd.concat([thinning_unit_frame(key, values, args.contract_root) for key in THINNING_UNITS], ignore_index=True)
    for dataset, frame in thinning.groupby("dataset", sort=False):
        rows += summarize(frame, thinning_statistics(), values, args.draws, args.seed, {"benchmark": "thinning", "dataset": dataset})
        print(f"thinning {dataset}: {len(frame)} thinned entries, {int(frame['selected'].sum())} filled at 5%", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "value_accuracy.csv", index=False, float_format="%.5f")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "values": values, "chosen": chosen, "draws": args.draws, "seed": args.seed,
        "masked_fill_fractions": list(FRACTIONS), "thinning_fill_fraction": THINNING_FILL,
    }, indent=2) + "\n")
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
