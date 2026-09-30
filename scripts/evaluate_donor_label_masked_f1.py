#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from masked_f1_units import UNIT_FRACTIONS, Unit, fraction_name, load_unit
from paired_masked_f1_bootstrap import pooled_f1, unit_arrays

REPOSITORY = Path(__file__).resolve().parents[1]
PANCREAS_TRUTH = REPOSITORY / "artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/preprocessed.h5ad"
METHODS = ("safe_fusion", "safe_fusion_donor")

def tissue_units(root: Path) -> dict[str, list[Unit]]:
    tissues: dict[str, list[Unit]] = {"pancreas": [], "colon": []}
    for fold in range(3):
        key = f"pancreas_{fold}"
        src = REPOSITORY / f"artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit/fold_{fold}"
        tissues["pancreas"].append(Unit(
            dataset="Pancreas", key=key, corrupted=src / "corrupted.h5ad", coordinates=src / "coordinates.parquet",
            splits=src / "splits.parquet", truth=PANCREAS_TRUTH, fit_split="development", unit_column="donor",
            tie_seed=1729, selector_dir=root / "masked_donor" / key,
        ))
    for fold in range(3):
        key = f"colon_{fold}"
        src = REPOSITORY / f"artifacts/paper_evidence/review_round3/colon_crossfit/fold_{fold}"
        tissues["colon"].append(Unit(
            dataset="Colon", key=key, corrupted=src / "corrupted.h5ad", coordinates=src / "coordinates.parquet",
            splits=src / "splits.parquet", truth=src / "prepared.h5ad", fit_split="validation", unit_column="donor",
            tie_seed=1729, selector_dir=root / "masked_donor" / key,
        ))
    return tissues

def _unit_frames(unit: Unit, data, rows, cols, labels) -> pd.DataFrame:
    frames = []
    for method in METHODS:
        selector_dir = unit.selector_dir / ("selector" if method == "safe_fusion" else "selector_donor")
        for fraction in UNIT_FRACTIONS:
            contract = selector_dir / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}"
            selected = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols]) != 0
            grouped = pd.DataFrame({"unit": data.unit_labels[rows], "selected": selected, "labels": labels})
            grouped["true_positive"] = grouped["selected"] & grouped["labels"].astype(bool)
            agg = grouped.groupby("unit", sort=True).agg(
                n_selected=("selected", "sum"), n_true_positive=("true_positive", "sum"), n_masked_positives=("labels", "sum"),
            ).reset_index()
            frames.append(agg.assign(dataset=unit.key, method=method, fraction=fraction))
    return pd.concat(frames, ignore_index=True)

def interval(point: float, boot: np.ndarray) -> list[float]:
    return [round(100 * point, 3), round(100 * float(np.quantile(boot, 0.025)), 3), round(100 * float(np.quantile(boot, 0.975)), 3)]

def tissue_report(units_frame: pd.DataFrame, draws: int, seed: int) -> dict:
    donors = sorted(units_frame["unit"].unique())
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(donors), size=(draws, len(donors)))
    arrays = {method: unit_arrays(frame, donors) for method, frame in units_frame.groupby("method", sort=False)}

    def mean_f1(counts, rows_=slice(None)) -> float:
        return float(pooled_f1(counts["n_selected"][rows_], counts["n_true_positive"][rows_], counts["n_masked_positives"][rows_]).mean())

    boot = {method: np.array([mean_f1(counts, draw) for draw in index]) for method, counts in arrays.items()}
    point = {method: mean_f1(counts) for method, counts in arrays.items()}
    return {
        "n_donors": len(donors),
        "mean_f1_percent": {method: interval(point[method], boot[method]) for method in METHODS},
        "donor_minus_labelfree_pp": interval(point["safe_fusion_donor"] - point["safe_fusion"], boot["safe_fusion_donor"] - boot["safe_fusion"]),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round4/transductive_references/sex_zeros")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    output = args.output_dir or (args.root / "masked_f1")
    output.mkdir(parents=True, exist_ok=True)

    tissues = tissue_units(args.root)
    all_records, report = [], {}
    for tissue, units in tissues.items():
        per_unit_frames = []
        for unit in units:
            data = load_unit(unit)
            test_cells = np.flatnonzero(data.split == "test")
            local_rows, cols = np.where(data.counts[test_cells] == 0)
            rows = test_cells[local_rows]
            labels = data.masked[rows, cols]
            per_unit_frames.append(_unit_frames(unit, data, rows, cols, labels))
            positives = int(labels.sum())
            for method in METHODS:
                selector_dir = unit.selector_dir / ("selector" if method == "safe_fusion" else "selector_donor")
                for fraction in UNIT_FRACTIONS:
                    contract = selector_dir / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}"
                    selected = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols]) != 0
                    true_positive = int((selected & labels).sum())
                    all_records.append({
                        "tissue": tissue, "unit": unit.key, "method": method, "fill_fraction": fraction,
                        "n_zeros": len(labels), "n_masked_positives": positives, "n_selected": int(selected.sum()),
                        "n_true_positive": true_positive,
                        "masked_f1": 2.0 * true_positive / (int(selected.sum()) + positives),
                    })
        units_frame = pd.concat(per_unit_frames, ignore_index=True)
        report[tissue] = tissue_report(units_frame, args.draws, args.seed)

    pd.DataFrame(all_records).to_csv(output / "masked_f1_by_fraction.csv", index=False)
    (output / "masked_f1_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))

if __name__ == "__main__":
    main()
