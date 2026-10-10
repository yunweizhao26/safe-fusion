#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from compute_matched_baseline_f1_curves import tie_broken_order
from fusion_value_bootstrap import STATISTICS, Group, group_statistics, interval, statistics_from_sums
from masked_f1_units import EVIDENCE, add_root_arguments, count_scale_values, load_unit, units_from_args

R3_ROOT = EVIDENCE / "review_round3" / "comparators"
FUSION_VALUE = EVIDENCE / "review_round2" / "fusion_value"
NEW_METHODS = ("DCA", "DCA (ZINB)", "scImpute", "scRecover", "EnImpute", "scVI (ZINB)")
PROBABILITY_METHODS = ("DCA (ZINB)", "scImpute", "scRecover", "scVI (ZINB)")
REFERENCES = ("Safe Fusion", "Safe Fusion (transductive)")

CONTEXT_SELECTORS = {"scVI (stacked)": "scvi_stacked", "MAGIC (stacked)": "magic_stacked", "SAVER (stacked)": "saver_stacked"}
CONTEXT_VALUES = ("SAVER", "MAGIC", "scVI")

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

def method_table() -> dict[str, dict]:
    table = {
        "Safe Fusion": {"family": "reference", "kind": "fusion_selector", "source": "safe_fusion"},
        "Safe Fusion (transductive)": {"family": "reference", "kind": "fusion_selector", "source": "safe_fusion_transductive"},
        "Random ranking": {"family": "reference", "kind": "constant", "source": None},
    }
    for name in CONTEXT_VALUES:
        table[name] = {"family": "paper comparator", "kind": "value", "source": name}
    table["scVI P(X>0)"] = {"family": "paper comparator", "kind": "nonzero_probability", "source": "scvi"}
    for name, source in CONTEXT_SELECTORS.items():
        table[name] = {"family": "paper selector", "kind": "fusion_selector", "source": source}
    for name in NEW_METHODS:
        table[name] = {"family": "new value", "kind": "value", "source": name}
        if name in PROBABILITY_METHODS:
            table[f"{name} P(dropout)"] = {"family": "new dropout probability", "kind": "dropout_probability", "source": name}
        table[f"{name} (stacked)"] = {"family": "new selector", "kind": "stacked", "source": slug(name)}
    return table

def scores_for(spec: dict, unit, data, rows: np.ndarray, cols: np.ndarray, args) -> tuple[np.ndarray, bool]:

    kind = spec["kind"]
    if kind == "constant":
        return np.zeros(len(rows)), False
    if kind == "fusion_selector":
        scores = np.load(args.fusion_selector_root / unit.key / spec["source"] / "test_scores.npy")
        return scores.astype(np.float64), True
    if kind == "stacked":
        stored = np.load(args.stacked_root / unit.key / spec["source"] / "test_scores.npz")
        if not (np.array_equal(stored["rows"], rows) and np.array_equal(stored["cols"], cols)):
            raise ValueError(f"{unit.key} {spec['source']}: stacked scores are in a different candidate order")
        return stored["score"].astype(np.float64), True
    if kind == "nonzero_probability":
        mean = np.load(args.probability_root / spec["source"] / unit.key / "mean.npy", mmap_mode="r")
        return np.asarray(mean[rows, cols], dtype=np.float64), False
    contract = unit.contracts[spec["source"]]
    if kind == "dropout_probability":
        probability = np.load(contract / "dropout_probability.npy", mmap_mode="r")
        return np.asarray(probability[rows, cols], dtype=np.float64), False
    values, _ = count_scale_values(contract, data, rows, cols)
    return values.astype(np.float64), False

def available(spec: dict, unit, args) -> Path | None:

    kind = spec["kind"]
    if kind == "constant":
        return None
    if kind == "fusion_selector":
        path = args.fusion_selector_root / unit.key / spec["source"] / "test_scores.npy"
    elif kind == "stacked":
        path = args.stacked_root / unit.key / spec["source"] / "test_scores.npz"
    elif kind == "nonzero_probability":
        path = args.probability_root / spec["source"] / unit.key / "mean.npy"
    elif kind == "dropout_probability":
        path = unit.contracts[spec["source"]] / "dropout_probability.npy"
    else:
        path = unit.contracts[spec["source"]] / "mean.npy"
    return None if path.exists() else path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_root_arguments(parser)
    parser.add_argument("--unit-keys", nargs="+", default=["pancreas_0", "pancreas_1", "pancreas_2", "colon", "norman_crispra"])
    parser.add_argument("--dataset-label", default=None, help="Replace the dataset name of every unit (for example Colon cross-fit).")
    parser.add_argument("--fusion-selector-root", type=Path, default=FUSION_VALUE / "selectors")
    parser.add_argument("--probability-root", type=Path, default=FUSION_VALUE / "nonzero_probability")
    parser.add_argument("--stacked-root", type=Path, default=R3_ROOT / "stacked")
    parser.add_argument("--output-dir", type=Path, default=R3_ROOT / "evaluation" / "masked")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    table = method_table()
    units = {unit.key: unit for unit in units_from_args(args, args.seed)}
    by_dataset: dict[str, list[str]] = {}
    for key in args.unit_keys:
        by_dataset.setdefault(args.dataset_label or units[key].dataset, []).append(key)

    absolute_rows, difference_rows, missing_rows, imputed_rows, count_frames = [], [], [], [], []
    for dataset, keys in by_dataset.items():
        loaded = []
        for key in keys:
            unit = units[key]
            data = load_unit(unit)
            rows, cols = np.where((data.counts == 0) & (data.split == "test")[:, None])
            loaded.append((unit, data, rows, cols, np.char.add(f"{key}:", data.unit_labels[rows].astype(str))))
        dataset_units = np.unique(np.concatenate([labels for *_, labels in loaded]))
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(dataset_units), size=(args.draws, len(dataset_units)))
        weights = np.zeros((args.draws + 1, len(dataset_units)))
        weights[0] = 1.0
        np.add.at(weights, (np.repeat(np.arange(1, args.draws + 1), len(dataset_units)), draws.ravel()), 1.0)

        per_method: dict[str, dict[str, np.ndarray]] = {}
        for method, spec in table.items():
            missing = [str(path) for unit, *_ in loaded if (path := available(spec, unit, args)) is not None]
            if missing:
                missing_rows.append({"dataset": dataset, "method": method, "missing": missing})
                print(json.dumps({"dataset": dataset, "method": method, "missing": missing}), flush=True)
                continue
            sums = None
            for unit, data, rows, cols, unit_labels in loaded:
                codes = np.searchsorted(dataset_units, unit_labels)
                group = Group(unit=unit, rows=rows, cols=cols, labels=data.masked[rows, cols], codes=codes, unit_index=np.unique(codes))
                scores, candidate_order = scores_for(spec, unit, data, rows, cols, args)
                if len(scores) != len(rows):
                    raise ValueError(f"{unit.key} {method}: {len(scores)} scores for {len(rows)} candidates")
                order = np.argsort(-scores, kind="stable") if candidate_order else tie_broken_order(scores, unit.tie_seed)
                rank = np.empty(len(order), dtype=np.int64)
                rank[order] = np.arange(len(order))
                if spec["kind"] in ("value", "dropout_probability"):
                    imputed_rows.append({
                        "dataset": dataset, "unit": unit.key, "method": method, "n_candidates": int(len(scores)),
                        "share_positive": float(np.mean(scores > 0)), "n_distinct": int(len(np.unique(scores))),
                    })
                result = group_statistics(rank, group, weights)
                result.pop("unit_counts")
                sums = result if sums is None else {name: sums[name] + result[name] for name in sums}
            per_method[method] = statistics_from_sums(sums)
            print(json.dumps({"dataset": dataset, "method": method, **{
                name: round(100 * float(values[0]), 3) for name, values in per_method[method].items()}}), flush=True)

        for method, statistics in per_method.items():
            for name, values in statistics.items():
                low, high = interval(values[1:])
                absolute_rows.append({"dataset": dataset, "method": method, "family": table[method]["family"], "statistic": name,
                                      "estimate_percent": 100 * values[0], "lower_percent": 100 * low, "upper_percent": 100 * high,
                                      "n_units": len(dataset_units)})
        for reference in REFERENCES:
            if reference not in per_method:
                continue
            for method, statistics in per_method.items():
                if method == reference:
                    continue
                for name in STATISTICS:
                    difference = per_method[reference][name] - statistics[name]
                    low, high = interval(difference[1:])
                    difference_rows.append({"dataset": dataset, "reference": reference, "comparator": method,
                                            "family": table[method]["family"], "statistic": name,
                                            "estimate_pp": 100 * difference[0], "lower_pp": 100 * low, "upper_pp": 100 * high,
                                            "n_units": len(dataset_units)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    absolute, differences = pd.DataFrame(absolute_rows), pd.DataFrame(difference_rows)
    absolute.to_csv(args.output_dir / "absolute.csv", index=False, float_format="%.4f")
    differences.to_csv(args.output_dir / "paired_differences.csv", index=False, float_format="%.4f")
    pd.DataFrame(imputed_rows).to_csv(args.output_dir / "score_support.csv", index=False, float_format="%.6f")
    (args.output_dir / "summary.json").write_text(json.dumps({
        "design": "Table 1 rule and paired bootstrap of fusion_value_bootstrap.py; see the module docstring.",
        "unit_keys": args.unit_keys, "draws": args.draws, "seed": args.seed, "references": list(REFERENCES),
        "missing": missing_rows,
    }, indent=2) + "\n")

if __name__ == "__main__":
    main()
