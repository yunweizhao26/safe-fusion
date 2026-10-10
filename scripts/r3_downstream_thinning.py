#!/usr/bin/env python3

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
sys.path.insert(0, str(REPOSITORY / "scripts"))

from disease_control_common import TISSUES, stratified_draws
from disease_control_effects import build_contrasts, prepared, run_tests

ROOT = REPOSITORY / "artifacts" / "paper_evidence" / "review_round3" / "downstream" / "thinning_reference"
DEPLOYMENT_TRAJECTORY = (
    REPOSITORY / "artifacts" / "paper_evidence" / "downstream_deployment" / "evaluation" / "trajectory" / "zebrafish" / "unit_metrics.parquet"
)
PANCREAS_TRUTH = REPOSITORY / "artifacts" / "pancreas_runs" / "0b2469810675-45c81b160d78" / "data" / "pancreas_islets" / "preprocessed.h5ad"
PANCREAS_THINNED = REPOSITORY / "artifacts" / "paper_evidence" / "thinning" / "data" / "pancreas_thinning_050" / "corrupted.h5ad"
METHODS = {"safe_fusion": "Safe Fusion", "svd": "SVD", "weighted_knn": "Weighted kNN", "magic": "MAGIC", "scvi": "scVI"}
PCTS = (1, 5, 10)
UNITS = {
    "Colon": ("colon_thinning_050",),
    "Pancreas": ("pancreas_thinning_050_0", "pancreas_thinning_050_1", "pancreas_thinning_050_2"),
    "Norman CRISPRa": ("norman_thinning_050",),
    "Zebrafish": ("zebrafish_thinning_050",),
}

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def method_names() -> list[str]:
    return [f"{method}_{pct}pct" for method in METHODS for pct in PCTS]

def split_name(name: str) -> tuple[str, int]:
    method, pct = name.rsplit("_", 1)
    return METHODS[method], int(pct.removesuffix("pct"))

def verdict(low: float, high: float) -> str:
    if low > 0:
        return "closer"
    if high < 0:
        return "farther"
    return "change only"

def unit_draws(strata: np.ndarray, draws: int, seed: int) -> np.ndarray:

    rng = np.random.default_rng(seed)
    levels = sorted(np.unique(strata))
    members = [np.flatnonzero(strata == level) for level in levels]
    return np.stack([
        np.concatenate([rng.choice(group, size=len(group), replace=True) for group in members])
        for _ in range(draws)
    ])

def unit_endpoints(root: Path, trajectory_reference: Path) -> pd.DataFrame:

    frames = []
    specs = [
        ("Colon", "Marker AUPRC", root / "evaluation/markers/colon/donor_metrics.parquet", "canonical_marker_pr_auc", "donor", "disease"),
        ("Colon", "Reference mapping F1", root / "evaluation/markers/colon/donor_metrics.parquet", "cell_identity_macro_f1", "donor", "disease"),
        ("Pancreas", "Marker AUPRC", root / "evaluation/pancreas_biology/donor_metrics.parquet", "canonical_marker_pr_auc", "donor", "condition"),
        ("Pancreas", "Reference mapping F1", root / "evaluation/pancreas_biology/donor_metrics.parquet", "cell_identity_macro_f1", "donor", "condition"),
        ("Zebrafish", "Dynamics", root / "evaluation/trajectory/zebrafish/unit_metrics.parquet", "dynamic_gene_spearman", "unit", None),
        ("Zebrafish", "Stage error", root / "evaluation/trajectory/zebrafish/unit_metrics.parquet", "stage_rank_mae", "unit", None),
        ("Norman CRISPRa", "Edge AUPRC", root / "evaluation/grn/norman_crispra/unit_metrics.parquet", "edge_pr_auc_q10", "unit", None),
    ]
    for dataset, endpoint, path, metric, unit_column, stratum_column in specs:
        if not path.exists():
            print(f"missing {path}; {dataset} {endpoint} skipped", flush=True)
            continue
        frame = pd.read_parquet(path)
        frame = frame[frame["metric"] == metric]
        pivot = frame.pivot(index=unit_column, columns="method", values="value")
        if endpoint == "Stage error":

            deployed = pd.read_parquet(trajectory_reference)
            deployed = deployed[(deployed["metric"] == metric) & (deployed["method"] == "corrupted_raw")]
            reference = deployed.set_index("unit")["value"].reindex(pivot.index)
        else:
            reference = pivot["reference_truth"]
        strata = (
            frame.drop_duplicates(unit_column).set_index(unit_column)[stratum_column].reindex(pivot.index).astype(str)
            if stratum_column else pd.Series("all", index=pivot.index)
        )
        for name in ["corrupted_raw", *method_names()]:
            if name not in pivot:
                raise ValueError(f"{path} lacks {name}")
            frames.append(pd.DataFrame({
                "dataset": dataset, "endpoint": endpoint, "unit": pivot.index.astype(str),
                "stratum": strata.to_numpy(), "matrix": name,
                "value": pivot[name].to_numpy(dtype=float), "unthinned": reference.to_numpy(dtype=float),
                "thinned_truth_pipeline": pivot["reference_truth"].to_numpy(dtype=float),
            }))
    return pd.concat(frames, ignore_index=True)

def summarize_units(values: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    rows = []
    for (dataset, endpoint), frame in values.groupby(["dataset", "endpoint"], sort=False):
        pivot = frame.pivot(index="unit", columns="matrix", values="value")
        units = pivot.index
        reference = frame.drop_duplicates("unit").set_index("unit")["unthinned"].reindex(units).to_numpy()
        strata = frame.drop_duplicates("unit").set_index("unit")["stratum"].reindex(units).to_numpy()
        index = unit_draws(strata, draws, seed)
        raw = pivot["corrupted_raw"].to_numpy()
        raw_error = np.abs(raw - reference)
        for name in method_names():
            filled = pivot[name].to_numpy()
            error = np.abs(filled - reference)
            reduction = raw_error - error
            change = filled - raw
            boot_reduction = np.nanmean(reduction[index], axis=1)
            boot_change = np.nanmean(change[index], axis=1)
            method, pct = split_name(name)
            low, high = np.nanquantile(boot_reduction, [0.025, 0.975])
            change_low, change_high = np.nanquantile(boot_change, [0.025, 0.975])
            rows.append({
                "dataset": dataset, "endpoint": endpoint, "method": method, "fill_pct": pct,
                "n_units": int(np.isfinite(reduction).sum()),
                "unthinned": float(np.nanmean(reference)),
                "thinned_unfilled": float(np.nanmean(raw)),
                "thinned_filled": float(np.nanmean(filled)),
                "change": float(np.nanmean(change)), "change_ci_low": float(change_low), "change_ci_high": float(change_high),
                "error_unfilled": float(np.nanmean(raw_error)), "error_filled": float(np.nanmean(error)),
                "error_reduction": float(np.nanmean(reduction)),
                "ci_low": float(low), "ci_high": float(high), "verdict": verdict(low, high),
            })
    return pd.DataFrame(rows)

def disease_correlation(root: Path) -> pd.DataFrame:

    summary = pd.read_parquet(root / "evaluation/pancreas_biology/bootstrap_summary.parquet")
    comparisons = pd.read_parquet(root / "evaluation/pancreas_biology/paired_comparisons.parquet")
    metric = "disease_logfc_spearman"
    summary = summary[(summary["scope"] == "disease_contrast") & (summary["metric"] == metric)]
    comparisons = comparisons[
        (comparisons["scope"] == "disease_contrast") & (comparisons["metric"] == metric)
        & (comparisons["reference"] == "corrupted_raw")
    ]
    rows = []
    for contrast, frame in summary.groupby("contrast", sort=True):
        estimates = frame.set_index("method")["estimate"]
        for name in method_names():
            method, pct = split_name(name)
            paired = comparisons[(comparisons["contrast"] == contrast) & (comparisons["method"] == name)].iloc[0]
            rows.append({
                "dataset": "Pancreas", "endpoint": f"Disease effects, {contrast.split('_')[0]}",
                "method": method, "fill_pct": pct, "n_units": int(paired["n_units"]),
                "unthinned": 1.0, "thinned_unfilled": float(estimates["corrupted_raw"]),
                "thinned_filled": float(estimates[name]),
                "change": float(paired["difference"]), "change_ci_low": float(paired["ci_low"]),
                "change_ci_high": float(paired["ci_high"]),
                "error_unfilled": float(1 - estimates["corrupted_raw"]), "error_filled": float(1 - estimates[name]),
                "error_reduction": float(paired["difference"]),
                "ci_low": float(paired["ci_low"]), "ci_high": float(paired["ci_high"]),
                "verdict": verdict(float(paired["ci_low"]), float(paired["ci_high"])),
            })
    return pd.DataFrame(rows)

def pseudobulk_effect(cube: np.ndarray, contrast, case: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    effect = np.zeros((len(contrast.cell_types), cube.shape[2]), dtype=np.float64)
    tested = np.zeros(len(contrast.cell_types), dtype=bool)
    for c in range(len(contrast.cell_types)):
        case_donors = [d for d in case if (d, c) in contrast.cells]
        control_donors = [d for d in control if (d, c) in contrast.cells]
        if len(case_donors) < 2 or len(control_donors) < 2:
            continue
        effect[c] = cube[case_donors, c].mean(axis=0) - cube[control_donors, c].mean(axis=0)
        tested[c] = True
    return effect, tested

def slope(reference: np.ndarray, other: np.ndarray, tested: np.ndarray) -> float:

    x = reference[tested].ravel()
    y = other[tested].ravel()
    return float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else float("nan")

def disease_slopes(root: Path, truth_path: Path, thinned_path: Path, draws: int, seed: int) -> pd.DataFrame:
    truth_adata = ad.read_h5ad(truth_path)
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    thinned_adata = ad.read_h5ad(thinned_path)
    if not np.array_equal(thinned_adata.obs_names, truth_adata.obs_names):
        raise ValueError("thinned and unthinned cell orders differ")
    thinned = dense(thinned_adata.layers["corrupted_counts"]).astype(np.float32)
    spec = TISSUES["pancreas"]
    cells = truth_adata.obs[["donor", "cell_type", spec["condition_column"]]].astype(str).rename(
        columns={spec["condition_column"]: "condition_label"}
    ).reset_index(drop=True)
    contrasts = build_contrasts(cells, spec)
    matrices = {"unthinned": truth, "corrupted_raw": thinned}
    for name in method_names():
        values = np.load(root / "evaluation/pancreas_biology" / f"oof_{name}.npy", allow_pickle=False)
        if np.isnan(values).any():
            raise ValueError(f"out-of-fold matrix of {name} is incomplete")
        matrices[name] = values
    rows = []
    for i, contrast in enumerate(contrasts):
        cubes = {name: prepared(matrix, contrast)["cube"] for name, matrix in matrices.items()}
        case = np.flatnonzero(contrast.is_case)
        control = np.flatnonzero(~contrast.is_case)
        observed = {name: pseudobulk_effect(cube, contrast, case, control) for name, cube in cubes.items()}
        check = run_tests(prepared(matrices["unthinned"], contrast), contrast, case, control, with_tests=False)["pseudobulk"]
        if not np.allclose(check[0], observed["unthinned"][0]) or not np.array_equal(check[3], observed["unthinned"][1]):
            raise AssertionError("pseudobulk effects differ from disease_control_effects.run_tests")
        tested = observed["unthinned"][1]
        slopes = {name: slope(observed["unthinned"][0], effect, tested) for name, (effect, _) in observed.items()}
        boot = {name: np.empty(draws) for name in matrices}
        for d, draw in enumerate(stratified_draws(contrast.is_case, draws, seed + i)):
            draw_case = draw[contrast.is_case[draw]]
            draw_control = draw[~contrast.is_case[draw]]
            effects = {name: pseudobulk_effect(cube, contrast, draw_case, draw_control) for name, cube in cubes.items()}
            draw_tested = effects["unthinned"][1]
            for name, (effect, _) in effects.items():
                boot[name][d] = slope(effects["unthinned"][0], effect, draw_tested)
        raw_error = np.abs(1 - boot["corrupted_raw"])
        for name in method_names():
            method, pct = split_name(name)
            reduction = raw_error - np.abs(1 - boot[name])
            change = boot[name] - boot["corrupted_raw"]
            low, high = np.nanquantile(reduction, [0.025, 0.975])
            change_low, change_high = np.nanquantile(change, [0.025, 0.975])
            rows.append({
                "dataset": "Pancreas", "endpoint": f"Disease effect slope, {contrast.name.split(' ')[0]}",
                "method": method, "fill_pct": pct, "n_units": int(len(contrast.donors)),
                "unthinned": 1.0, "thinned_unfilled": slopes["corrupted_raw"], "thinned_filled": slopes[name],
                "change": slopes[name] - slopes["corrupted_raw"],
                "change_ci_low": float(change_low), "change_ci_high": float(change_high),
                "error_unfilled": abs(1 - slopes["corrupted_raw"]), "error_filled": abs(1 - slopes[name]),
                "error_reduction": abs(1 - slopes["corrupted_raw"]) - abs(1 - slopes[name]),
                "ci_low": float(low), "ci_high": float(high), "verdict": verdict(float(low), float(high)),
            })
    return pd.DataFrame(rows)

def fill_precision(root: Path) -> pd.DataFrame:

    rows = []
    for dataset, units in UNITS.items():
        totals: dict[str, list[int]] = {}
        prevalence = [0, 0]
        for key in units:
            unit = root / "units" / key
            manifest = json.loads((unit / "safe_fusion_1pct" / "metadata.json").read_text())
            source = Path(manifest["parameters"]["deployment"]["source_contract"])
            models = source.parents[1]
            split = pd.read_parquet(unit / "splits.parquet").set_index("cell_id").loc[manifest["cell_ids"], "split"].to_numpy()
            test = split == "test"
            hybrid = dense(ad.read_h5ad(models / "input" / "hybrid.h5ad").layers["corrupted_counts"]).astype(np.float32)[test]
            coordinates = pd.read_parquet(models / "input" / "coordinates.parquet")
            positive = np.zeros(hybrid.shape, dtype=bool)
            rows_test = np.cumsum(test) - 1
            thinning = coordinates[test[coordinates["cell_index"].to_numpy(dtype=int)]]
            positive[rows_test[thinning["cell_index"].to_numpy(dtype=int)], thinning["gene_index"].to_numpy(dtype=int)] = True
            zeros = hybrid == 0
            prevalence[0] += int((positive & zeros).sum())
            prevalence[1] += int(zeros.sum())
            for name in method_names():
                filled = np.load(unit / name / "mean.npy", mmap_mode="r")[test] != hybrid
                filled &= zeros
                count = totals.setdefault(name, [0, 0])
                count[0] += int((filled & positive).sum())
                count[1] += int(filled.sum())
        for name, (hits, selected) in totals.items():
            method, pct = split_name(name)
            rows.append({
                "dataset": dataset, "method": method, "fill_pct": pct, "n_filled": selected,
                "precision": hits / selected if selected else float("nan"),
                "prevalence": prevalence[0] / prevalence[1],
                "recall": hits / prevalence[0] if prevalence[0] else float("nan"),
            })
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--unthinned-trajectory", type=Path, default=DEPLOYMENT_TRAJECTORY)
    parser.add_argument("--pancreas-truth", type=Path, default=PANCREAS_TRUTH)
    parser.add_argument("--pancreas-thinned", type=Path, default=PANCREAS_THINNED)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else REPOSITORY / args.root
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)

    values = unit_endpoints(root, args.unthinned_trajectory)
    values.to_csv(output / "unit_endpoint_values.csv", index=False)
    tables = [summarize_units(values, args.draws, args.seed)]
    if (root / "evaluation/pancreas_biology/paired_comparisons.parquet").exists():
        tables.append(disease_correlation(root))
        tables.append(disease_slopes(root, args.pancreas_truth, args.pancreas_thinned, args.draws, args.seed))
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(output / "error_reduction.csv", index=False)
    precision = fill_precision(root)
    precision.to_csv(output / "fill_precision.csv", index=False)
    counts = result.groupby(["method", "fill_pct", "verdict"]).size().unstack(fill_value=0)
    counts.to_csv(output / "verdict_counts.csv")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(result[["dataset", "endpoint", "method", "fill_pct", "unthinned", "thinned_unfilled", "thinned_filled",
                      "error_reduction", "ci_low", "ci_high", "verdict"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(counts.to_string())
        print(precision.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

if __name__ == "__main__":
    main()
