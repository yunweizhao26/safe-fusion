#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import load_units_manifest
from selector_attribution import exact_topk
from v2_value_models import TRAINING_SPLITS, fit_values, load_benchmark, predict_values, selector_detection

EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"
TRANSFER = EVIDENCE / "review_round2" / "thinning_transfer"
THINNING = EVIDENCE / "thinning"
MANIFEST = EVIDENCE / "review_round2" / "leakage_free" / "units_manifest.json"
COLON_CROSSFIT = EVIDENCE / "review_round3" / "colon_crossfit"
COLON_MANIFEST = COLON_CROSSFIT / "units_manifest.json"
OUTPUT = EVIDENCE / "review_round3" / "value_v2_ablations" / "value" / "inner"
DATASETS = {"pancreas": "Pancreas", "colon": "Colon", "norman": "CRISPRa"}
THINNING_UNITS = {
    "colon_thinning_050": THINNING / "data" / "colon_thinning_050",
    "pancreas_thinning_050_0": THINNING / "data" / "pancreas_thinning_050",
    "pancreas_thinning_050_1": THINNING / "data" / "pancreas_thinning_050",
    "pancreas_thinning_050_2": THINNING / "data" / "pancreas_thinning_050",
    "norman_thinning_050": TRANSFER / "data" / "norman_thinning_050",
    **{f"colon_thinning_050_{fold}": COLON_CROSSFIT / "thinning" / "data" / f"colon_thinning_050_{fold}" for fold in range(3)},
}

THINNING_ROOTS = {key: (COLON_CROSSFIT / "thinning" if key.startswith("colon_thinning_050_") else TRANSFER) for key in THINNING_UNITS}
MASKED_UNITS = ("pancreas_0", "pancreas_1", "pancreas_2", "colon", "norman_crispra", "colon_0", "colon_1", "colon_2")
UNITS = (*THINNING_UNITS, *MASKED_UNITS)

@dataclass
class UnitSpec:
    key: str
    kind: str
    dataset: str
    input: Path
    coordinates: Path
    splits: Path
    teacher_root: Path
    fit_split: str
    unit_column: str
    thinned_dir: Path | None

def fold_label(key: str) -> str | None:

    head, _, last = key.rpartition("_")
    return f"fold_{last}" if head and last.isdigit() and len(last) == 1 else None

def unit_spec(key: str, manifest: Path, colon_manifest: Path = COLON_MANIFEST) -> UnitSpec:
    dataset = DATASETS[key.split("_")[0]]
    unit_column = "target" if dataset == "CRISPRa" else "donor"
    if key in THINNING_UNITS:
        root = THINNING_ROOTS[key] / "mask_trained" / key
        return UnitSpec(
            key=key, kind="thinning", dataset=dataset, input=root / "input" / "hybrid.h5ad",
            coordinates=root / "input" / "coordinates.parquet", splits=root / "input" / "splits.parquet",
            teacher_root=root, fit_split="validation" if dataset == "Colon" else "development",
            unit_column=unit_column, thinned_dir=THINNING_UNITS[key],
        )
    units = {entry.key: entry for path in (manifest, colon_manifest) for entry in load_units_manifest(path)}
    unit = units[key]
    return UnitSpec(
        key=key, kind="masked", dataset=dataset, input=unit.corrupted, coordinates=unit.coordinates,
        splits=unit.splits, teacher_root=unit.contracts["Gene median"].parent, fit_split=unit.fit_split,
        unit_column=unit.unit_column, thinned_dir=None,
    )

def inner_test_cells(groups: np.ndarray, training: np.ndarray, by_cells: bool, fraction: float, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    inner = np.zeros(len(groups), dtype=bool)
    names = np.array(sorted(set(groups[training].tolist())))
    if not by_cells:
        chosen = rng.permutation(names)[: int(round(fraction * len(names)))]
        return training & np.isin(groups, chosen)
    for name in names:
        rows = np.flatnonzero(training & (groups == name))
        inner[rng.permutation(rows)[: int(round(fraction * len(rows)))]] = True
    return inner

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", choices=UNITS, help="Unit key, or use --array-index.")
    parser.add_argument("--array-index", type=int, default=None, help="Index into the unit list.")
    parser.add_argument("--units-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--colon-manifest", type=Path, default=COLON_MANIFEST)
    parser.add_argument("--inner-fraction", type=float, default=0.20)
    parser.add_argument("--fill-fraction", type=float, default=0.05)
    parser.add_argument("--retained-fraction", type=float, default=0.5, help="Retained fraction of the thinning units.")
    parser.add_argument("--mask-rate", type=float, default=0.10, help="Design masking rate of the training cells.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    key = UNITS[args.array_index] if args.array_index is not None else args.unit
    if key is None:
        parser.error("give --unit or --array-index")
    spec = unit_spec(key, args.units_manifest, args.colon_manifest)
    started = time.perf_counter()

    bench = load_benchmark(spec.input, spec.coordinates, spec.splits, spec.teacher_root)
    training = bench.training
    groups = bench.obs[spec.unit_column].astype(str).to_numpy()
    inner_test = inner_test_cells(groups, training, spec.dataset == "CRISPRa", args.inner_fraction, args.seed)
    inner_train = training & ~inner_test
    fitted = fit_values(bench, inner_train, args.mask_rate, args.seed)

    rows, cols = np.nonzero((bench.counts == 0) & inner_test[:, None])
    selector_cells = (bench.split == spec.fit_split) & inner_train
    score, probability, detection, detection_report = selector_detection(
        bench, selector_cells, rows, cols, args.mask_rate, args.seed,
    )
    selected = exact_topk(score, max(1, int(round(args.fill_fraction * len(score)))))
    n_genes = bench.counts.shape[1]
    candidate_key = rows.astype(np.int64) * n_genes + cols

    frames = []
    entry_sets = [("masked", *np.nonzero(bench.masked & inner_test[:, None]))]
    if spec.kind == "thinning":
        thinning = pd.read_parquet(spec.thinned_dir / "coordinates.parquet")
        thin_rows = thinning["cell_index"].to_numpy(dtype=np.int64)
        thin_cols = thinning["gene_index"].to_numpy(dtype=np.int64)
        keep = inner_test[thin_rows]
        if np.any(bench.counts[thin_rows[keep], thin_cols[keep]] != 0):
            raise ValueError("thinned entries of the inner test cells are not zero in the input")
        original = np.zeros(bench.counts.shape, dtype=np.float32)
        original[thin_rows[keep], thin_cols[keep]] = thinning["original_value"].to_numpy(dtype=np.float32)[keep]
        entry_sets.append(("thinned", thin_rows[keep], thin_cols[keep]))
    for entry_type, entry_rows, entry_cols in entry_sets:
        location = np.searchsorted(candidate_key, entry_rows.astype(np.int64) * n_genes + entry_cols)
        if not np.array_equal(candidate_key[location], entry_rows.astype(np.int64) * n_genes + entry_cols):
            raise ValueError(f"{entry_type} entries are not candidate zeros of the inner test cells")
        values = predict_values(fitted, bench, entry_rows, entry_cols, detection[location])
        if entry_type == "thinned":
            count = original[entry_rows, entry_cols]
            expected = args.retained_fraction * count
        else:
            count = bench.hidden[entry_rows, entry_cols]
            expected = np.full(len(entry_rows), np.nan, dtype=np.float32)
        fold = fold_label(key)
        group = groups[entry_rows] if fold is None else np.char.add(f"{fold}:", groups[entry_rows].astype(str))
        frames.append(pd.DataFrame({
            "dataset": spec.dataset, "unit": key, "benchmark": spec.kind, "entry_type": entry_type,
            "group": group, "cell_index": entry_rows.astype(np.int32), "gene_index": entry_cols.astype(np.int32),
            "count": count, "expected_count": expected, "selected": selected[location],
            "selector_score": score[location], "detection": detection[location],
            **{f"value:{name}": value for name, value in values.items()},
        }))
    entries = pd.concat(frames, ignore_index=True)
    output = args.output_root / key
    output.mkdir(parents=True, exist_ok=True)
    entries.to_parquet(output / "entries.parquet", index=False)
    report = {
        "unit": key, "kind": spec.kind, "dataset": spec.dataset,
        "paths": {name: str(getattr(spec, name)) for name in ("input", "coordinates", "splits", "teacher_root", "thinned_dir")},
        "fit_split": spec.fit_split, "unit_column": spec.unit_column,
        "inner_fraction": args.inner_fraction, "fill_fraction": args.fill_fraction, "mask_rate": args.mask_rate,
        "retained_fraction": args.retained_fraction if spec.kind == "thinning" else None, "seed": args.seed,
        "training_cells": int(training.sum()), "inner_test_cells": int(inner_test.sum()),
        "inner_train_cells": int(inner_train.sum()), "selector_cells": int(selector_cells.sum()),
        "inner_test_groups": sorted(set(groups[inner_test].tolist())),
        "inner_test_candidates": int(len(rows)), "selected_candidates": int(selected.sum()),
        "saved_entries": entries.groupby("entry_type").size().to_dict(),
        "saved_entries_selected": entries.groupby("entry_type")["selected"].sum().astype(int).to_dict(),
        "value_models": fitted.report, "detection": detection_report,
        "held_out_test_cells_used": False, "splits_used": list(TRAINING_SPLITS),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({key: report["saved_entries"], "selected": report["saved_entries_selected"],
                      "theta": fitted.report["negative_binomial_dispersion"], "seconds": round(report["elapsed_seconds"])}))

if __name__ == "__main__":
    main()
