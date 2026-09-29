#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from evaluate_thinning_transfer import DESIGNS, datasets
from masked_f1_units import EVIDENCE, fraction_name

TEACHERS = ("gene_median", "svd_impute", "graph_smooth", "magic_inductive", "scvi_inductive")
PRODUCTION = EVIDENCE / "downstream_deployment"
OUTPUT = EVIDENCE / "review_round2" / "inserted_value"
DEPLOYMENT = {
    "Pancreas": [PRODUCTION / f"pancreas/fold_{fold}" for fold in range(3)],
    "Colon": [PRODUCTION / "colon"],
    "CRISPRa (Norman)": [OUTPUT / "deployment/norman_crispra"],
    "CRISPRi (Adamson)": [PRODUCTION / "adamson_crispri"],
    "Knockout (Dixit)": [PRODUCTION / "dixit_ko"],
    "ECCITE-seq (Papalexi)": [PRODUCTION / "papalexi_eccite"],
    "Zebrafish": [PRODUCTION / "zebrafish"],
}
FILLS = {"Safe Fusion": "selector/safe_fusion_calibrated_mlp_topk_{}", "SVD": "matched/svd_topk_{}",
         "Weighted kNN": "matched/weighted_knn_topk_{}"}
FRACTIONS = (0.01, 0.05, 0.10)
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
EXPECTED_THRESHOLDS = (2.0, 5.0)
RETAINED = {"colon_thinning_050": 0.5, "colon_thinning_025": 0.25, "pancreas_thinning_050": 0.5, "norman_thinning_050": 0.5}
VALUES = {"Boosted fused value": "safe_fusion", "Linear combination": "safe_fusion_linear", "scVI teacher": "scvi_inductive"}
COUNT_BINS = [0, 1, 3, 10, np.inf]
COUNT_LABELS = ["1", "2-3", "4-10", ">10"]


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def at(path: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.asarray(np.load(path, mmap_mode="r")[rows, cols], dtype=np.float64)


def distribution(values: np.ndarray, expected: np.ndarray) -> dict:
    row = {"n": int(len(values))}
    if len(values):
        row.update({f"value_q{int(100 * q):02d}": float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))})
        row.update({"value_mean": float(values.mean()), "value_above_1_share": float(np.mean(values > 1.0))})
    row["expected_median"] = float(np.median(expected))
    row.update({f"expected_above_{t:g}_share": float(np.mean(expected > t)) for t in EXPECTED_THRESHOLDS})
    return row


def recorded_zero_fills() -> pd.DataFrame:
    rows_out = []
    for dataset, units in DEPLOYMENT.items():
        expected_all, filled = [], {(name, b): ([], []) for name in FILLS for b in FRACTIONS}
        for unit in units:
            recorded = ad.read_h5ad(unit / "recorded.h5ad")
            counts = dense(recorded.layers["corrupted_counts"])
            split = pd.read_parquet(unit / "splits.parquet").set_index("cell_id").loc[recorded.obs_names.astype(str), "split"].to_numpy()
            test = np.flatnonzero(split == "test")
            local_rows, cols = np.where(counts[test] == 0)
            rows = test[local_rows]
            expected = np.mean([np.maximum(at(unit / "methods" / name / "mean.npy", rows, cols), 0.0) for name in TEACHERS], axis=0)
            expected_all.append(expected)
            for name, pattern in FILLS.items():
                for b in FRACTIONS:
                    value = at(unit / pattern.format(fraction_name(b)) / "mean.npy", rows, cols)
                    chosen = value != 0
                    target = max(1, int(round(b * len(rows))))
                    if abs(int(chosen.sum()) - target) > 1:
                        raise ValueError(f"{unit}: {name} fills {int(chosen.sum())} zeros at {b}, expected {target}")
                    filled[(name, b)][0].append(value[chosen])
                    filled[(name, b)][1].append(expected[chosen])
        expected_all = np.concatenate(expected_all)
        rows_out.append({"dataset": dataset, "method": "All recorded zeros", "fill_fraction": None,
                         **distribution(np.empty(0), expected_all)})
        for (name, b), (values, expected) in filled.items():
            rows_out.append({"dataset": dataset, "method": name, "fill_fraction": b,
                             **distribution(np.concatenate(values), np.concatenate(expected))})
    return pd.DataFrame(rows_out)


def thinning_positive_bias(transfer_root: Path, thinning_root: Path, draws: int, seed: int) -> pd.DataFrame:
    specs = datasets(transfer_root, thinning_root)
    rows_out = []
    for key, dataset in specs.items():
        data = ad.read_h5ad(dataset["data"] / "corrupted.h5ad")
        truth = ad.read_h5ad(dataset["truth"])
        thinned = dense(data.layers["corrupted_counts"])
        truth_counts = dense(truth.layers["counts"])
        cell_ids = data.obs_names.astype(str).to_numpy()
        bio_units = truth.obs[dataset["unit_column"]].astype(str).to_numpy()
        frames = []
        for unit in dataset["units"]:
            split = pd.read_parquet(unit["splits"]).set_index("cell_id").loc[cell_ids, "split"].to_numpy()
            test = split == "test"
            rows, cols = np.where((thinned == 0) & (truth_counts > 0) & test[:, None])
            frame = pd.DataFrame({"unit": unit["key"] + ":" + pd.Series(bio_units[rows]).astype(str),
                                  "recorded": truth_counts[rows, cols].astype(np.float64)})
            for design in DESIGNS:
                root = unit["thin_root"] if design == "thinning-trained" else unit["mask_root"]
                filled = at(root / f"selector/safe_fusion_calibrated_mlp_topk_{fraction_name(0.05)}" / "mean.npy", rows, cols)
                frame[f"selected:{design}"] = filled != 0
                for label, contract in VALUES.items():
                    frame[f"{label}:{design}"] = np.maximum(at(root / contract / "mean.npy", rows, cols), 0.0)
            frames.append(frame)
        frame = pd.concat(frames, ignore_index=True)
        frame["count"] = pd.cut(np.round(frame["recorded"]), COUNT_BINS, labels=COUNT_LABELS).astype(str)
        units = sorted(frame["unit"].unique())
        index = np.random.default_rng(seed).integers(0, len(units), size=(draws, len(units)))
        references = {"recorded count": np.log1p(frame["recorded"].to_numpy()),
                      "expected thinned count": np.log1p(RETAINED[key] * frame["recorded"].to_numpy())}
        for design in DESIGNS:
            for subset in ("all positives", "filled at 5%"):
                keep = np.ones(len(frame), bool) if subset == "all positives" else frame[f"selected:{design}"].to_numpy()
                for stratum in ["all", *COUNT_LABELS]:
                    mask = keep & ((frame["count"] == stratum).to_numpy() if stratum != "all" else True)
                    if not mask.any():
                        continue
                    group = pd.Categorical(frame["unit"][mask], categories=units)
                    n = np.bincount(group.codes, minlength=len(units)).astype(np.float64)
                    row = {"dataset": key, "retained_fraction": RETAINED[key], "design": design, "positives": subset,
                           "recorded_count": stratum, "n": int(mask.sum()),
                           "recorded_mean": float(frame["recorded"][mask].mean())}
                    for label in VALUES:
                        value = frame[f"{label}:{design}"].to_numpy()[mask]
                        row[f"{label}:inserted_mean"] = float(value.mean())
                        for reference, target in references.items():
                            error = np.log1p(value) - target[mask]
                            sums = np.bincount(group.codes, weights=error, minlength=len(units))
                            drawn = n[index].sum(axis=1)
                            sample = sums[index].sum(axis=1)[drawn > 0] / drawn[drawn > 0]
                            name = f"{label}:bias_vs_{reference.replace(' ', '_')}"
                            row[name] = float(error.mean())
                            row[f"{name}_low"] = float(np.quantile(sample, 0.025))
                            row[f"{name}_high"] = float(np.quantile(sample, 0.975))
                        row[f"{label}:abs_error_vs_recorded_count"] = float(np.abs(np.log1p(value) - references["recorded count"][mask]).mean())
                    rows_out.append(row)
        print(f"{key}: {len(frame)} held-out thinning positives", flush=True)
    return pd.DataFrame(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=["recorded", "thinning", "all"])
    parser.add_argument("--transfer-root", type=Path, default=EVIDENCE / "review_round2" / "thinning_transfer")
    parser.add_argument("--thinning-root", type=Path, default=EVIDENCE / "thinning")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.part in ("recorded", "all"):
        table = recorded_zero_fills()
        table.to_csv(args.output_dir / "recorded_zero_fills.csv", index=False)
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(table.round(3).to_string(index=False))
    if args.part in ("thinning", "all"):
        table = thinning_positive_bias(args.transfer_root, args.thinning_root, args.draws, args.seed)
        table.to_csv(args.output_dir / "thinning_positive_bias.csv", index=False)
        columns = ["dataset", "design", "positives", "recorded_count", "n",
                   *[f"{label}:bias_vs_{ref}" for label in VALUES for ref in ("recorded_count", "expected_thinned_count")]]
        with pd.option_context("display.width", 300, "display.max_columns", 30):
            print(table[columns].round(3).to_string(index=False))
    (args.output_dir / "settings.json").write_text(json.dumps({
        "fill_fractions": FRACTIONS, "value_quantiles": QUANTILES, "expected_count": "mean of the five teacher proposals",
        "expected_count_thresholds": EXPECTED_THRESHOLDS, "retained_fraction": RETAINED, "draws": args.draws, "seed": args.seed,
    }, indent=1) + "\n")


if __name__ == "__main__":
    main()
