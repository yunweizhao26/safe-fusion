#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_matched_baseline_f1_curves import tie_broken_order
from masked_f1_units import EVIDENCE, ROOT, Unit, count_scale_values, fraction_name, load_unit
from selector_attribution import exact_topk

TEACHERS = ("Gene median", "SVD", "Weighted kNN", "MAGIC (inductive)", "scVI (inductive)")
RANKED = ("SVD", "MAGIC", "scVI")
STACKED = {"Selector on scVI": "scvi", "Selector on MAGIC": "magic"}
COUNT_BINS = [0, 1, 3, 10, np.inf]
COUNT_LABELS = ["1", "2-3", "4-10", ">10"]
QUINTILES = 5
OUTPUT = EVIDENCE / "review_round2" / "thinning_transfer" / "count_stratified_recall"
UNITS_MANIFEST = EVIDENCE / "review_round2" / "leakage_free" / "units_manifest.json"
PATH_FIELDS = ("corrupted", "coordinates", "splits", "truth", "selector_dir")


def manifest_units(path: Path) -> list[Unit]:
    units = []
    for entry in json.loads(path.read_text()):
        fields = {key: ROOT / value if key in PATH_FIELDS else value for key, value in entry.items()}
        fields["contracts"] = {name: ROOT / value for name, value in entry["contracts"].items()}
        units.append(Unit(**fields))
    return units


def unit_frame(unit, stacked_root: Path, fraction: float) -> pd.DataFrame:
    data = load_unit(unit)
    test_cells = np.flatnonzero(data.split == "test")
    local_rows, cols = np.where(data.counts[test_cells] == 0)
    rows = test_cells[local_rows]
    labels = data.masked[rows, cols]
    n = len(labels)
    k = max(1, int(round(fraction * n)))
    selected = {}
    for name in RANKED:
        scores, _ = count_scale_values(unit.contracts[name], data, rows, cols)
        order = tie_broken_order(scores, unit.tie_seed)
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        selected[name] = rank < k
    filled = np.asarray(np.load(unit.selector_dir / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}" / "mean.npy",
                                mmap_mode="r")[rows, cols])
    selected["Safe Fusion"] = filled != 0
    if int(selected["Safe Fusion"].sum()) != k:
        raise ValueError(f"{unit.key}: Safe Fusion fills {int(selected['Safe Fusion'].sum())} zeros, expected {k}")
    for name, directory in STACKED.items():
        saved = np.load(stacked_root / unit.key / directory / "test_scores.npz")
        if not (np.array_equal(saved["rows"], rows) and np.array_equal(saved["cols"], cols)):
            raise ValueError(f"{unit.key}: stacked scores of {name} are not on the held-out candidate zeros")
        selected[name] = exact_topk(saved["score"], k)

    coordinates = pd.read_parquet(unit.coordinates)
    original = np.zeros(data.counts.shape, dtype=np.float32)
    original[coordinates["cell_index"].to_numpy(dtype=int), coordinates["gene_index"].to_numpy(dtype=int)] = coordinates["original_value"].to_numpy()
    positive_rows, positive_cols = rows[labels], cols[labels]
    expected = np.mean([count_scale_values(unit.contracts[name], data, positive_rows, positive_cols)[0] for name in TEACHERS], axis=0)
    frame = pd.DataFrame({
        "dataset": unit.dataset,
        "unit": unit.key + ":" + pd.Series(data.unit_labels[positive_rows]).astype(str),
        "original": original[positive_rows, positive_cols],
        "expected": expected,
    })
    if np.any(frame["original"] <= 0):
        raise ValueError(f"{unit.key}: a held-out masked positive has no original count")
    for name, mask in selected.items():
        frame[name] = mask[labels]
    return frame


def stratum_table(frame: pd.DataFrame, stratum: str, methods: list[str], draws: int, seed: int) -> list[dict]:
    units = sorted(frame["unit"].unique())
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(units), size=(draws, len(units)))
    rows = []
    for label, part in frame.groupby(stratum, sort=True, observed=True):
        grouped = part.groupby("unit")
        positives = grouped.size().reindex(units, fill_value=0).to_numpy(dtype=np.float64)
        hits = {name: grouped[name].sum().reindex(units, fill_value=0).to_numpy(dtype=np.float64) for name in methods}
        row = {"stratum": stratum, "level": str(label), "n_positives": int(positives.sum()),
               "share_of_positives": float(len(part) / len(frame)),
               "expected_count_median": float(part["expected"].median()),
               "original_count_median": float(part["original"].median())}
        for name in methods:
            row[f"recall:{name}"] = 100 * hits[name].sum() / positives.sum()
        resampled_positives = positives[index].sum(axis=1)
        valid = resampled_positives > 0
        for name in methods:
            if name == "Safe Fusion":
                continue
            difference = (hits["Safe Fusion"][index].sum(axis=1) - hits[name][index].sum(axis=1))[valid] / resampled_positives[valid]
            row[f"sf_minus:{name}"] = row["recall:Safe Fusion"] - row[f"recall:{name}"]
            row[f"sf_minus_low:{name}"] = 100 * float(np.quantile(difference, 0.025))
            row[f"sf_minus_high:{name}"] = 100 * float(np.quantile(difference, 0.975))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units-manifest", type=Path, default=UNITS_MANIFEST)
    parser.add_argument("--stacked-root", type=Path, default=OUTPUT / "stacked")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--fraction", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    frames = [unit_frame(unit, args.stacked_root, args.fraction) for unit in manifest_units(args.units_manifest)]
    methods = ["Safe Fusion", *RANKED, *STACKED]
    tables, edges = [], {}
    for dataset, frame in pd.concat(frames, ignore_index=True).groupby("dataset", sort=False):
        frame = frame.copy()
        frame["original_count"] = pd.cut(np.round(frame["original"]), COUNT_BINS, labels=COUNT_LABELS)
        quantiles = np.quantile(frame["expected"], np.linspace(0, 1, QUINTILES + 1))
        edges[dataset] = quantiles.tolist()
        frame["expected_count_quintile"] = np.clip(np.searchsorted(quantiles, frame["expected"], side="right"), 1, QUINTILES)
        overall = {name: 100 * float(frame[name].mean()) for name in methods}
        print(f"== {dataset}: {len(frame)} held-out masked positives; overall recall {json.dumps({k: round(v, 2) for k, v in overall.items()})}")
        for stratum in ("original_count", "expected_count_quintile"):
            rows = stratum_table(frame, stratum, methods, args.draws, args.seed)
            for row in rows:
                row["dataset"] = dataset
            tables.extend(rows)
    table = pd.DataFrame(tables)
    table.insert(0, "dataset", table.pop("dataset"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "count_stratified_recall.csv", index=False)
    (args.output_dir / "count_stratified_recall.json").write_text(json.dumps({
        "fill_fraction": args.fraction, "draws": args.draws, "seed": args.seed,
        "expected_count": "mean of the five teacher proposals on the count scale",
        "expected_count_quintile_edges": edges, "rows": tables,
    }, indent=1, default=float) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(table[["dataset", "stratum", "level", "n_positives", "expected_count_median",
                     *[f"recall:{m}" for m in methods]]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
