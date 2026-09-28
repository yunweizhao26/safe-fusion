#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from masked_f1_units import fraction_name

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT = REPOSITORY / "artifacts" / "paper_evidence" / "value_accuracy"

BOOSTED = "safe_fusion"
LINEAR = "safe_fusion_linear"
TEACHERS = ("gene_median", "svd_impute", "graph_smooth", "magic_inductive", "scvi_inductive")
VALUES = (BOOSTED, LINEAR, "autoencoder_fusion_3teachers", "autoencoder_fusion_5teachers", *TEACHERS)
FRACTIONS = tuple(round(0.01 * step, 2) for step in range(1, 11))
UNIT_KEYS = {
    "pancreas_0": ("pancreas", "donor"),
    "pancreas_1": ("pancreas", "donor"),
    "pancreas_2": ("pancreas", "donor"),
    "colon": ("colon", "donor"),
    "norman_crispra": ("norman_crispra", "target"),
}
TRUE_COUNT_BINS = [0, 1, 3, 10, np.inf]
TRUE_COUNT_LABELS = ["1", "2-3", "4-10", ">10"]
DETECTION_GROUPS = 5
UNIT_FIELDS = ("input", "coordinates", "splits", "methods_root", "selector")


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def held_out_positives(spec: dict) -> dict:
    adata = ad.read_h5ad(spec["input"])
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = dense(matrix).astype(np.float32)
    split = pd.read_parquet(spec["splits"]).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    coordinates = pd.read_parquet(spec["coordinates"])
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    keep = (split[rows] == "test") & (counts[rows, cols] == 0)
    original = coordinates["original_value"].to_numpy(dtype=np.float64)[keep]
    if np.any(original <= 0):
        raise ValueError(f"held-out positives of {spec['coordinates']} include a zero true count")
    return {
        "obs": adata.obs,
        "counts": counts,
        "training": np.isin(split, ["development", "validation"]),
        "rows": rows[keep],
        "cols": cols[keep],
        "original": original,
        "y": np.log1p(original),
    }


def log_value(contract: Path, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    value = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols], dtype=np.float64)
    return np.log1p(np.maximum(value, 0.0))


def selected_zeros(selector: Path, fraction: float, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    filled = selector / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}" / "mean.npy"
    return np.asarray(np.load(filled, mmap_mode="r")[rows, cols]) > 0


def table_unit(key: str, spec: dict) -> pd.DataFrame:
    dataset, unit_column = UNIT_KEYS[key]
    data = held_out_positives(spec)
    rows, cols, y = data["rows"], data["cols"], data["y"]
    detection = (data["counts"][data["training"]] > 0).mean(axis=0)
    detection_group = np.minimum(
        (pd.Series(detection).rank(pct=True).to_numpy() * DETECTION_GROUPS).astype(int), DETECTION_GROUPS - 1
    )
    columns = {
        "dataset": dataset,
        "unit": key,
        "bio_unit": data["obs"][unit_column].astype(str).to_numpy()[rows],
        "y": y,
        "detection_quintile": detection_group[cols] + 1,
        "true_count": pd.cut(np.round(data["original"]), TRUE_COUNT_BINS, labels=TRUE_COUNT_LABELS).astype(str),
    }
    errors = {value: np.abs(log_value(Path(spec["methods_root"]) / value, rows, cols) - y) for value in VALUES}
    columns.update(errors)
    for fraction in FRACTIONS:
        selected = selected_zeros(Path(spec["selector"]), fraction, rows, cols)
        for value in VALUES:
            columns[f"{fraction}:{value}"] = np.where(selected, errors[value], y)
    print(f"{key}: {len(y)} held-out positives", flush=True)
    return pd.DataFrame(columns)


def bootstrap_interval(numerator: np.ndarray, denominator: np.ndarray, draws: np.ndarray) -> tuple[float, float, float]:
    resampled = numerator[draws].sum(axis=1) / denominator[draws].sum(axis=1)
    return float(numerator.sum() / denominator.sum()), float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))


def dataset_tables(dataset: str, frame: pd.DataFrame, draws: np.ndarray) -> tuple[list, list, list]:
    groups = frame.groupby("bio_unit")
    sums = groups[list(VALUES)].sum()
    positives = groups.size().loc[sums.index].to_numpy(dtype=np.float64)
    input_error = groups["y"].sum().loc[sums.index].to_numpy()
    log_error = []
    for value in VALUES:
        difference, low, high = bootstrap_interval(sums[value].to_numpy() - sums[LINEAR].to_numpy(), positives, draws)
        log_error.append({
            "dataset": dataset, "value": value, "n_positives": int(positives.sum()),
            "log_error": float(frame[value].mean()),
            "minus_linear": difference, "minus_linear_low": low, "minus_linear_high": high,
        })
    error_removed = []
    for fraction in FRACTIONS:
        inserted = groups[[f"{fraction}:{value}" for value in VALUES]].sum().loc[sums.index]
        removed = {value: 100 * (input_error - inserted[f"{fraction}:{value}"].to_numpy()) for value in VALUES}
        for value in VALUES:
            difference, low, high = bootstrap_interval(removed[BOOSTED] - removed[value], input_error, draws)
            error_removed.append({
                "dataset": dataset, "fill_fraction": fraction, "value": value,
                "error_removed": float(removed[value].sum() / input_error.sum()),
                "boosted_minus": difference, "boosted_minus_low": low, "boosted_minus_high": high,
            })
    strata = []
    for kind in ("true_count", "detection_quintile"):
        for stratum, part in frame.groupby(kind):
            for value in VALUES:
                strata.append({
                    "dataset": dataset, "stratum_kind": kind, "stratum": str(stratum),
                    "n_positives": len(part), "value": value, "log_error": float(part[value].mean()),
                })
    return log_error, error_removed, strata


def cross_validation(key: str, spec: dict) -> list[dict]:
    parameters = json.loads((Path(spec["methods_root"]) / BOOSTED / "metadata.json").read_text())["parameters"]
    return [
        {
            "unit": key, "value_model": model, "out_of_fold_log_error": result["out_of_fold_log1p_mae"],
            **{f"fold_{fold + 1}_log_error": error for fold, error in enumerate(result["fold_log1p_mae"])},
        }
        for model, result in parameters["value_model_cross_validation"].items()
    ]


def replicate_row(label: str, spec: dict, fraction: float) -> dict:
    data = held_out_positives(spec)
    rows, cols, y = data["rows"], data["cols"], data["y"]
    selected = selected_zeros(Path(spec["selector"]), fraction, rows, cols)
    row = {"unit": label, "n_positives": len(y)}
    for name, value in (("boosted", BOOSTED), ("linear", LINEAR)):
        error = np.abs(log_value(Path(spec["methods_root"]) / value, rows, cols) - y)
        row[f"log_error_{name}"] = float(error.mean())
        row[f"error_removed_{name}"] = float(100 * (y.sum() - np.where(selected, error, y).sum()) / y.sum())
    row["log_error_boosted_minus_linear"] = row["log_error_boosted"] - row["log_error_linear"]
    row["error_removed_boosted_minus_linear"] = row["error_removed_boosted"] - row["error_removed_linear"]
    print(f"{label}: {len(y)} held-out positives", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--unit", nargs=6, action="append", default=[], metavar=("KEY", *(f.upper() for f in UNIT_FIELDS)),
        help="A unit of Table 2: dataset key of unit_paths.sh, input, coordinates, splits, "
             "directory of the value contracts, and selector output directory.",
    )
    parser.add_argument(
        "--replicate", nargs=6, action="append", default=[], metavar=("LABEL", *(f.upper() for f in UNIT_FIELDS)),
        help="A replicate unit (label benchmark/unit); the directory must hold safe_fusion and safe_fusion_linear.",
    )
    parser.add_argument("--replicate-fill-fraction", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    units = {spec[0]: dict(zip(UNIT_FIELDS, spec[1:])) for spec in args.unit}
    frame = pd.concat([table_unit(key, spec) for key, spec in units.items()], ignore_index=True)
    rng = np.random.default_rng(args.seed)
    log_error, error_removed, strata = [], [], []
    for dataset in sorted(frame["dataset"].unique()):
        part = frame[frame["dataset"] == dataset]
        n_units = part["bio_unit"].nunique()
        draws = rng.integers(0, n_units, size=(args.bootstrap, n_units))
        tables = dataset_tables(dataset, part, draws)
        log_error += tables[0]
        error_removed += tables[1]
        strata += tables[2]
    outputs = {
        "log_error.csv": pd.DataFrame(log_error),
        "error_removed.csv": pd.DataFrame(error_removed),
        "log_error_strata.csv": pd.DataFrame(strata),
        "cross_validation.csv": pd.DataFrame([row for key, spec in units.items() for row in cross_validation(key, spec)]),
    }
    replicates = pd.DataFrame([
        replicate_row(spec[0], dict(zip(UNIT_FIELDS, spec[1:])), args.replicate_fill_fraction) for spec in args.replicate
    ])
    summary = {
        "bootstrap": args.bootstrap, "seed": args.seed, "fill_fractions": list(FRACTIONS),
        "replicate_fill_fraction": args.replicate_fill_fraction, "units": units,
    }
    if len(replicates):
        outputs["replicates.csv"] = replicates
        benchmark = replicates["unit"].str.split("/").str[0]
        summary["replicates"] = {
            name: {
                "units": len(part),
                "boosted_removes_more_error": int((part["error_removed_boosted_minus_linear"] > 0).sum()),
                "boosted_lower_log_error": int((part["log_error_boosted_minus_linear"] < 0).sum()),
            }
            for name, part in replicates.groupby(benchmark)
        }
    for name, table in outputs.items():
        table.to_csv(args.output_dir / name, index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary.get("replicates", {}), indent=2))


if __name__ == "__main__":
    main()
