#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from compute_matched_baseline_f1_curves import tie_broken_order
from masked_f1_units import EVIDENCE, ROOT, UNIT_FRACTIONS, Unit, count_scale_values, fraction_name, load_unit, unit_counts
from paired_masked_f1_bootstrap import pooled_f1, unit_arrays

SCREENS = ("norman_crispra", "adamson_crispri", "dixit_ko", "papalexi_eccite")
VALUE_METHODS = {
    "svd": "svd_impute",
    "weighted_knn": "graph_smooth",
    "magic": "magic_inductive",
    "scvi": "scvi_inductive",
    "knn_condition": "graph_smooth_condition",
    "scvi_condition": "scvi_inductive_condition",
}
SELECTOR_METHODS = ("safe_fusion", "safe_fusion_condition")
METHOD_ORDER = ("safe_fusion", "svd", "weighted_knn", "magic", "scvi",
                "safe_fusion_condition", "knn_condition", "scvi_condition")
LABEL_PAIRS = (("safe_fusion_condition", "safe_fusion"), ("knn_condition", "weighted_knn"), ("scvi_condition", "scvi"))
LABEL_CONTRACTS = ("knn_condition", "scvi_condition")


@dataclass
class NormanPaths:
    benchmark: Path
    methods: Path
    label_methods: Path
    truth: Path
    selector: Path


def screen_unit(dataset: str, external: Path, norman: NormanPaths, seed: int) -> tuple[Unit, dict[str, Path]]:
    if dataset == "norman_crispra":
        root, methods, label_methods, truth = norman.benchmark, norman.methods, norman.label_methods, norman.truth
        selectors = {"safe_fusion": norman.selector, "safe_fusion_condition": label_methods / "selector_condition"}
    else:
        root = methods = label_methods = external / dataset
        selectors = {
            "safe_fusion": root / "selector_mlp_biology_range",
            "safe_fusion_condition": root / "selector_condition",
        }
        truth = ROOT / "external_data" / "prepared" / f"{dataset}.h5ad"
    unit = Unit(
        dataset=dataset,
        key=dataset,
        corrupted=root / "corrupted.h5ad",
        coordinates=root / "coordinates.parquet",
        splits=root / "splits.parquet",
        truth=truth,
        fit_split="development",
        unit_column="target",
        tie_seed=seed,
        selector_dir=selectors["safe_fusion"],
        contracts={name: (label_methods if name in LABEL_CONTRACTS else methods) / contract for name, contract in VALUE_METHODS.items()},
    )
    return unit, selectors


def evaluate_screen(unit: Unit, selectors: dict[str, Path]) -> tuple[list[dict], pd.DataFrame]:
    data = load_unit(unit)
    test_cells = np.flatnonzero(data.split == "test")
    local_rows, cols = np.where(data.counts[test_cells] == 0)
    rows = test_cells[local_rows]
    labels = data.masked[rows, cols]
    n_zeros, positives = len(labels), int(labels.sum())

    selections: dict[str, dict[float, np.ndarray]] = {}
    for name, contract in unit.contracts.items():
        scores, _ = count_scale_values(contract, data, rows, cols)
        rank = np.empty(n_zeros, dtype=np.int64)
        rank[tie_broken_order(scores, unit.tie_seed)] = np.arange(n_zeros)
        selections[name] = {fraction: rank < max(1, int(round(fraction * n_zeros))) for fraction in UNIT_FRACTIONS}
    for name in SELECTOR_METHODS:
        selections[name] = {
            fraction: np.asarray(
                np.load(selectors[name] / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}" / "mean.npy", mmap_mode="r")[rows, cols]
            ) != 0
            for fraction in UNIT_FRACTIONS
        }

    records, units = [], []
    for name in METHOD_ORDER:
        for fraction, selected in selections[name].items():
            units.append(unit_counts(selected, labels, data.unit_labels[rows]).assign(dataset=unit.dataset, method=name, fraction=fraction))
            true_positive = int((selected & labels).sum())
            records.append({
                "dataset": unit.dataset,
                "method": name,
                "fill_fraction": fraction,
                "n_zeros": n_zeros,
                "n_masked_positives": positives,
                "n_selected": int(selected.sum()),
                "n_true_positive": true_positive,
                "masked_f1": 2.0 * true_positive / (int(selected.sum()) + positives),
            })
    return records, pd.concat(units, ignore_index=True)


def interval(point: float, boot: np.ndarray) -> list[float]:
    return [round(100 * point, 3), round(100 * float(np.quantile(boot, 0.025)), 3), round(100 * float(np.quantile(boot, 0.975)), 3)]


def paired_screen_intervals(units: pd.DataFrame, draws: int, seed: int) -> dict:
    report = {}
    for dataset, frame in units.groupby("dataset", sort=False):
        labels = sorted(frame["unit"].unique())
        rng = np.random.default_rng(seed)
        index = rng.integers(0, len(labels), size=(draws, len(labels)))
        arrays = {method: unit_arrays(method_frame, labels) for method, method_frame in frame.groupby("method", sort=False)}

        def mean_f1(counts: dict, rows: np.ndarray | slice = slice(None)) -> float:
            return float(pooled_f1(counts["n_selected"][rows], counts["n_true_positive"][rows], counts["n_masked_positives"][rows]).mean())

        boot = {method: np.array([mean_f1(counts, draw) for draw in index]) for method, counts in arrays.items()}
        entry = {"n_units": len(labels), "mean_f1_percent": {method: interval(mean_f1(arrays[method]), boot[method]) for method in METHOD_ORDER}}
        entry["label_minus_unlabeled_pp"] = {
            f"{label}_minus_{plain}": interval(mean_f1(arrays[label]) - mean_f1(arrays[plain]), boot[label] - boot[plain])
            for label, plain in LABEL_PAIRS
        }
        report[dataset] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE)
    parser.add_argument("--external-root", type=Path, default=ROOT / "artifacts" / "external_perturbseq")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE / "perturbation_zeros")
    parser.add_argument("--unit-output-dir", type=Path, default=None,
                        help="When given, also write masked F1 per screen with intervals over perturbation labels.")
    parser.add_argument("--norman-benchmark", type=Path, default=None,
                        help="Defaults to <evidence-root>/review_round2/leakage_free/norman_crispra.")
    parser.add_argument("--norman-methods", type=Path, default=None, help="Defaults to <norman-benchmark>/methods.")
    parser.add_argument("--norman-label-methods", type=Path, default=None, help="Defaults to <norman-methods>.")
    parser.add_argument("--norman-truth", type=Path, default=None, help="Defaults to <norman-benchmark>/prepared.h5ad.")
    parser.add_argument("--norman-selector", type=Path, default=None,
                        help="Defaults to <evidence-root>/review_round2/leakage_free/selector_mlp_biology_range_fullteachers/norman_crispra.")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    leakage_free = args.evidence_root / "review_round2" / "leakage_free"
    benchmark = args.norman_benchmark or leakage_free / "norman_crispra"
    methods = args.norman_methods or benchmark / "methods"
    norman = NormanPaths(
        benchmark=benchmark,
        methods=methods,
        label_methods=args.norman_label_methods or methods,
        truth=args.norman_truth or benchmark / "prepared.h5ad",
        selector=args.norman_selector or leakage_free / "selector_mlp_biology_range_fullteachers" / "norman_crispra",
    )

    records, unit_frames = [], []
    for dataset in SCREENS:
        unit, selectors = screen_unit(dataset, args.external_root, norman, args.seed)
        screen_records, screen_units = evaluate_screen(unit, selectors)
        records.extend(screen_records)
        unit_frames.append(screen_units)
    frame = pd.DataFrame(records)
    by_screen = frame.groupby(["method", "dataset"], sort=False)["masked_f1"].mean().unstack("dataset")[list(SCREENS)]
    by_screen = by_screen.loc[list(METHOD_ORDER)]
    report = {
        "fill_fractions": list(UNIT_FRACTIONS),
        "masked_f1_mean_over_fractions_by_screen": by_screen.round(6).to_dict(orient="index"),
        "masked_f1_mean_over_screens": by_screen.mean(axis=1).round(6).to_dict(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "masked_f1_by_fraction.csv", index=False)
    (args.output_dir / "masked_f1_report.json").write_text(json.dumps(report, indent=1) + "\n")
    table = (100 * by_screen).assign(mean_over_screens=100 * by_screen.mean(axis=1))
    print(table.round(2).to_string())

    if args.unit_output_dir is None:
        return
    units = pd.concat(unit_frames, ignore_index=True)
    args.unit_output_dir.mkdir(parents=True, exist_ok=True)
    units.to_csv(args.unit_output_dir / "masked_f1_unit_counts.csv", index=False)
    paired = paired_screen_intervals(units, args.draws, args.seed)
    (args.unit_output_dir / "masked_f1_by_screen.json").write_text(json.dumps(paired, indent=1) + "\n")
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
