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
from scipy.spatial.distance import cdist

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.downstream import randomized_pca_embedding
from safefusion_benchmark.marker_panels import PANELS
from safefusion_benchmark.metrics import average_precision_tie_aware, log1p_mae, spearman

DISEASE_MARKERS = {"CXCL10", "STAT1", "B2M", "IFITM1", "IFITM3"}

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def parse_named_subdir(value: str) -> tuple[str, str]:
    name, subdir = value.split("=", 1)
    if not name or not subdir:
        raise ValueError("extra methods must use name=subdirectory")
    return name, subdir

def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    values: list[float] = []
    for label in np.unique(actual):
        tp = np.sum((actual == label) & (predicted == label))
        fp = np.sum((actual != label) & (predicted == label))
        fn = np.sum((actual == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        values.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(values))

def centroid_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> np.ndarray:
    embedding = randomized_pca_embedding(matrix, train, seed=seed, components=30)
    classes = np.unique(labels[train])
    centroids = np.stack([embedding[train & (labels == label)].mean(axis=0) for label in classes])
    return classes[np.argmin(cdist(embedding[test], centroids), axis=1)]

def canonical_marker_metrics(
    matrix: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    gene_lookup: dict[str, int],
    panel: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    average_precisions: list[float] = []
    true_effects: list[float] = []
    predicted_effects: list[float] = []
    ectopic_filled = 0
    ectopic_total = 0
    for marker, targets in panel.items():
        if marker not in gene_lookup:
            continue
        gene = gene_lookup[marker]
        target = np.isin(labels[rows], np.asarray(targets, dtype=str))
        if not target.any() or target.all():
            continue
        true_values = truth[rows, gene]
        predicted_values = matrix[rows, gene]
        average_precisions.append(average_precision_tie_aware(target, predicted_values))
        true_effects.append(float(np.log1p(true_values[target].mean()) - np.log1p(true_values[~target].mean())))
        predicted_effects.append(float(np.log1p(predicted_values[target].mean()) - np.log1p(predicted_values[~target].mean())))
        ectopic = (~target) & (true_values == 0)
        ectopic_filled += int(np.sum(predicted_values[ectopic] > 0.5))
        ectopic_total += int(np.sum(ectopic))
    return {
        "canonical_marker_pr_auc": float(np.mean(average_precisions)),
        "canonical_marker_effect_spearman": spearman(np.asarray(true_effects), np.asarray(predicted_effects)),
        "ectopic_marker_fill_rate": float(ectopic_filled / ectopic_total) if ectopic_total else float("nan"),
    }

def development_marker_metrics(
    matrix: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    rows: np.ndarray,
) -> dict[str, float]:
    average_precisions: list[float] = []
    correlations: list[float] = []
    for label in np.unique(labels[train]):
        in_train = train & (labels == label)
        out_train = train & (labels != label)
        target = labels[rows] == label
        if not in_train.any() or not out_train.any() or not target.any() or target.all():
            continue
        reference_effect = np.log1p(truth[in_train].mean(axis=0)) - np.log1p(truth[out_train].mean(axis=0))
        markers = reference_effect >= np.quantile(reference_effect, 0.9)
        predicted_effect = np.log1p(matrix[rows[target]].mean(axis=0)) - np.log1p(matrix[rows[~target]].mean(axis=0))
        average_precisions.append(average_precision_tie_aware(markers, predicted_effect))
        correlations.append(spearman(reference_effect, predicted_effect))
    return {
        "development_marker_pr_auc": float(np.mean(average_precisions)),
        "development_marker_rank_spearman": float(np.mean(correlations)),
    }

def stratified_bootstrap_indices(frame: pd.DataFrame, replicates: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    result: list[np.ndarray] = []
    condition = frame["condition"].astype(str).to_numpy()
    for _ in range(replicates):
        selected: list[int] = []
        for level in sorted(np.unique(condition)):
            positions = np.flatnonzero(condition == level)
            selected.extend(rng.choice(positions, size=len(positions), replace=True).tolist())
        result.append(np.asarray(selected, dtype=int))
    return result

def summarize_unit_metrics(
    unit_metrics: pd.DataFrame,
    bootstrap_indices: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimates: list[dict] = []
    comparisons: list[dict] = []
    methods = sorted(unit_metrics["method"].unique())
    metrics = sorted(unit_metrics["metric"].unique())
    donor_order = unit_metrics[["donor", "condition"]].drop_duplicates().sort_values("donor")
    for metric in metrics:
        pivot = (
            unit_metrics[unit_metrics["metric"] == metric]
            .pivot(index="donor", columns="method", values="value")
            .reindex(donor_order["donor"])
        )
        bootstrap_values: dict[str, np.ndarray] = {}
        for method in methods:
            values = pivot[method].to_numpy(dtype=float)
            samples = np.asarray([np.nanmean(values[index]) for index in bootstrap_indices])
            bootstrap_values[method] = samples
            estimates.append({
                "scope": "donor",
                "contrast": "all_conditions",
                "method": method,
                "metric": metric,
                "estimate": float(np.nanmean(values)),
                "ci_low": float(np.nanquantile(samples, 0.025)),
                "ci_high": float(np.nanquantile(samples, 0.975)),
                "n_units": int(np.sum(np.isfinite(values))),
            })
        for method in methods:
            if method in {"reference_truth", "corrupted_raw"}:
                continue
            for reference in ("corrupted_raw", "graph_smooth"):
                if reference not in bootstrap_values or method == reference:
                    continue
                difference = bootstrap_values[method] - bootstrap_values[reference]
                observed = float(np.nanmean(pivot[method] - pivot[reference]))
                comparisons.append({
                    "scope": "donor",
                    "contrast": "all_conditions",
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "difference": observed,
                    "ci_low": float(np.nanquantile(difference, 0.025)),
                    "ci_high": float(np.nanquantile(difference, 0.975)),
                    "n_units": int(np.sum(np.isfinite(pivot[method] - pivot[reference]))),
                })
    return pd.DataFrame(estimates), pd.DataFrame(comparisons)

def pseudobulk_logcpm(
    matrix: np.ndarray,
    donors: np.ndarray,
    cell_types: np.ndarray,
    donor_levels: list[str],
    cell_type_levels: list[str],
) -> np.ndarray:
    result = np.full((len(donor_levels), len(cell_type_levels), matrix.shape[1]), np.nan, dtype=np.float32)
    for donor_index, donor in enumerate(donor_levels):
        for cell_index, cell_type in enumerate(cell_type_levels):
            selected = (donors == donor) & (cell_types == cell_type)
            if not selected.any():
                continue
            total = matrix[selected].sum(axis=0, dtype=np.float64)
            library = float(total.sum())
            if library > 0:
                result[donor_index, cell_index] = np.log1p(total * (1e6 / library)).astype(np.float32)
    return result

def contrast_metrics(reference_effect: np.ndarray, predicted_effect: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(reference_effect) & np.isfinite(predicted_effect)
    truth = reference_effect[valid]
    predicted = predicted_effect[valid]
    strong = np.abs(truth) >= np.quantile(np.abs(truth), 0.9)
    return {
        "disease_logfc_rmse": float(np.sqrt(np.mean(np.square(predicted - truth)))),
        "disease_logfc_spearman": spearman(truth, predicted),
        "disease_direction_top_decile": float(np.mean(np.sign(truth[strong]) == np.sign(predicted[strong]))),
    }

def summarize_disease_contrasts(
    matrices: dict[str, np.ndarray],
    truth: np.ndarray,
    donors: np.ndarray,
    conditions: np.ndarray,
    cell_types: np.ndarray,
    feature_names: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    donor_levels = sorted(np.unique(donors).tolist())
    donor_condition = pd.DataFrame({"donor": donors, "condition": conditions}).drop_duplicates().set_index("donor")
    condition_by_donor = donor_condition.loc[donor_levels, "condition"].astype(str).to_numpy()
    cell_type_levels = sorted(np.unique(cell_types).tolist())
    all_matrices = {"reference_truth": truth, **matrices}
    pseudobulks = {
        method: pseudobulk_logcpm(matrix, donors, cell_types, donor_levels, cell_type_levels)
        for method, matrix in all_matrices.items()
    }
    gene_lookup = {gene: index for index, gene in enumerate(feature_names)}
    disease_gene_indices = np.asarray([gene_lookup[gene] for gene in sorted(DISEASE_MARKERS & set(gene_lookup))], dtype=int)
    rng = np.random.default_rng(seed)
    summaries: list[dict] = []
    comparisons: list[dict] = []
    marker_rows: list[dict] = []
    for case in ("T1D", "AAB"):
        case_positions = np.flatnonzero(condition_by_donor == case)
        control_positions = np.flatnonzero(condition_by_donor == "Control")
        if len(case_positions) < 2 or len(control_positions) < 2:
            continue

        eligible_cells = []
        for cell_index in range(len(cell_type_levels)):
            case_present = np.sum(np.isfinite(pseudobulks["reference_truth"][case_positions, cell_index, 0]))
            control_present = np.sum(np.isfinite(pseudobulks["reference_truth"][control_positions, cell_index, 0]))
            if case_present >= 2 and control_present >= 2:
                eligible_cells.append(cell_index)
        eligible = np.asarray(eligible_cells, dtype=int)
        if not len(eligible):
            raise ValueError(f"no cell types support donor-level {case} versus Control inference")

        def effect(cube: np.ndarray, cases: np.ndarray, controls: np.ndarray) -> np.ndarray:
            return np.nanmean(cube[cases][:, eligible], axis=0) - np.nanmean(cube[controls][:, eligible], axis=0)

        observed_truth = effect(pseudobulks["reference_truth"], case_positions, control_positions)
        observed_by_method = {
            method: contrast_metrics(observed_truth, effect(cube, case_positions, control_positions))
            for method, cube in pseudobulks.items()
            if method != "reference_truth"
        }
        for method, cube in pseudobulks.items():
            if method == "reference_truth":
                continue
            predicted_effect = effect(cube, case_positions, control_positions)
            for cell_local, cell_index in enumerate(eligible):
                for gene_index in disease_gene_indices:
                    marker_rows.append({
                        "contrast": f"{case}_vs_Control",
                        "method": method,
                        "cell_type": cell_type_levels[cell_index],
                        "gene": feature_names[gene_index],
                        "truth_logcpm_difference": float(observed_truth[cell_local, gene_index]),
                        "predicted_logcpm_difference": float(predicted_effect[cell_local, gene_index]),
                    })

        bootstrap: dict[str, dict[str, list[float]]] = {
            method: {metric: [] for metric in next(iter(observed_by_method.values()))}
            for method in observed_by_method
        }
        for _ in range(replicates):
            sampled_cases = rng.choice(case_positions, size=len(case_positions), replace=True)
            sampled_controls = rng.choice(control_positions, size=len(control_positions), replace=True)
            truth_effect = effect(pseudobulks["reference_truth"], sampled_cases, sampled_controls)
            for method, cube in pseudobulks.items():
                if method == "reference_truth":
                    continue
                values = contrast_metrics(truth_effect, effect(cube, sampled_cases, sampled_controls))
                for metric, value in values.items():
                    bootstrap[method][metric].append(value)

        for method, metrics in observed_by_method.items():
            for metric, observed in metrics.items():
                samples = np.asarray(bootstrap[method][metric], dtype=float)
                summaries.append({
                    "scope": "disease_contrast",
                    "contrast": f"{case}_vs_Control",
                    "method": method,
                    "metric": metric,
                    "estimate": observed,
                    "ci_low": float(np.nanquantile(samples, 0.025)),
                    "ci_high": float(np.nanquantile(samples, 0.975)),
                    "n_units": int(len(case_positions) + len(control_positions)),
                    "n_case": int(len(case_positions)),
                    "n_control": int(len(control_positions)),
                    "n_cell_types": int(len(eligible)),
                })
        comparison_references = ["corrupted_raw", "graph_smooth"] + sorted(
            method for method in observed_by_method
            if method.startswith("logistic_")
        )
        for method in observed_by_method:
            if method == "corrupted_raw":
                continue
            for reference in comparison_references:
                if method == reference:
                    continue
                for metric in observed_by_method[method]:
                    difference = np.asarray(bootstrap[method][metric]) - np.asarray(bootstrap[reference][metric])
                    comparisons.append({
                        "scope": "disease_contrast",
                        "contrast": f"{case}_vs_Control",
                        "method": method,
                        "reference": reference,
                        "metric": metric,
                        "difference": observed_by_method[method][metric] - observed_by_method[reference][metric],
                        "ci_low": float(np.nanquantile(difference, 0.025)),
                        "ci_high": float(np.nanquantile(difference, 0.975)),
                        "n_units": int(len(case_positions) + len(control_positions)),
                    })
    return pd.DataFrame(summaries), pd.DataFrame(comparisons), pd.DataFrame(marker_rows)

def main() -> None:
    import time as _time

    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--crossfit-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--safe-fusion-subdir", default="safe_fusion")
    parser.add_argument("--extra-method", action="append", default=[], help="name=fold_subdirectory")
    parser.add_argument("--marker-panel", choices=["source", "original"], default="source")
    parser.add_argument(
        "--allow-transductive",
        action="store_true",
        help="accept contracts that declare a transductive fit on all cells (standard MAGIC and scVI) and mark them in the leakage table",
    )
    args = parser.parse_args()
    panel = PANELS["pancreas"][args.marker_panel]

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    labels = truth_adata.obs["cell_type"].astype(str).to_numpy()
    broad_labels = truth_adata.obs["cell_type_broad"].astype(str).to_numpy()
    donors = truth_adata.obs["donor"].astype(str).to_numpy()
    conditions = truth_adata.obs["condition"].astype(str).to_numpy()
    feature_names = truth_adata.var["feature_name"].astype(str).to_numpy()
    gene_lookup = {gene: index for index, gene in enumerate(feature_names)}
    coordinates = pd.read_parquet(args.coordinates)
    crossfit = Path(args.crossfit_dir)
    print(f"phase=loaded elapsed={_time.time():.0f}", flush=True)

    learned_subdirs = {
        "graph_smooth": "graph_smooth",
        "safe_fusion": args.safe_fusion_subdir,
        **dict(parse_named_subdir(value) for value in args.extra_method),
    }

    oof = {
        "corrupted_raw": corrupted.copy(),
        **{method: np.full_like(corrupted, np.nan) for method in learned_subdirs},
    }
    donor_records: list[dict] = []
    heldout_donors: set[str] = set()
    leakage_checks: list[dict] = []
    for fold in range(args.folds):
        split_path = crossfit / f"fold_{fold}" / "splits.parquet"
        split_frame = pd.read_parquet(split_path).set_index("cell_id").loc[truth_adata.obs_names.astype(str)]
        test = split_frame["split"].to_numpy() == "test"
        train = ~test
        fold_donors = set(donors[test])
        if heldout_donors & fold_donors:
            raise ValueError("a donor appears in multiple held-out folds")
        heldout_donors |= fold_donors
        matrices = {
            "reference_truth": truth,
            "corrupted_raw": corrupted,
            **{
                method: np.load(crossfit / f"fold_{fold}" / subdir / "mean.npy", allow_pickle=False)
                for method, subdir in learned_subdirs.items()
            },
        }
        for method, method_subdir in learned_subdirs.items():
            oof[method][test] = matrices[method][test]
            metadata = json.loads((crossfit / f"fold_{fold}" / method_subdir / "metadata.json").read_text())
            parameters = metadata.get("parameters", {})
            decision = parameters.get("decision_layer", {})
            decision_passed = (
                not decision
                or decision.get("test_values_used_for_thresholds") is False
                or (
                    decision.get("budget_mode") == "apply_topk"
                    and decision.get("test_labels_used_for_thresholds") is False
                )
            )
            leakage_checks.append({
                "fold": fold,
                "method": method,
                "training_cells": parameters.get("fit_cells"),
                "heldout_cells": parameters.get("heldout_test_cells"),
                "test_used_for_fit": parameters.get("test_used_for_fit"),
                "passed": bool(parameters.get("test_used_for_fit") is False and parameters.get("fit_cells") == int(train.sum()) and decision_passed),
            })
            if args.allow_transductive:
                transductive = parameters.get("transductive") is True and not decision
                leakage_checks[-1]["transductive"] = transductive
                leakage_checks[-1]["passed"] = leakage_checks[-1]["passed"] or transductive

        coordinate_rows = coordinates["cell_index"].to_numpy(dtype=int)
        fold_coordinates = coordinates[np.isin(coordinate_rows, np.flatnonzero(test))]
        for method, matrix in matrices.items():
            detailed_prediction = centroid_predictions(matrix, labels, train, test, args.seed + fold)
            broad_prediction = centroid_predictions(matrix, broad_labels, train, test, args.seed + 100 + fold)
            for donor in sorted(fold_donors):
                rows = np.flatnonzero(test & (donors == donor))
                local_test = np.flatnonzero(np.isin(np.flatnonzero(test), rows))
                metrics = {
                    "cell_identity_macro_f1": macro_f1(labels[rows], detailed_prediction[local_test]),
                    "broad_cell_identity_macro_f1": macro_f1(broad_labels[rows], broad_prediction[local_test]),
                    **canonical_marker_metrics(matrix, truth, labels, rows, gene_lookup, panel),
                    **development_marker_metrics(matrix, truth, labels, train, rows),
                }
                donor_coordinates = fold_coordinates[fold_coordinates["cell_index"].isin(rows)]
                marker_mask = feature_names[donor_coordinates["gene_index"].to_numpy(dtype=int)]
                donor_coordinates = donor_coordinates[np.isin(marker_mask, sorted(panel))]
                if len(donor_coordinates):
                    coordinate_row = donor_coordinates["cell_index"].to_numpy(dtype=int)
                    coordinate_gene = donor_coordinates["gene_index"].to_numpy(dtype=int)
                    metrics["masked_canonical_marker_log1p_mae"] = log1p_mae(
                        donor_coordinates["original_value"].to_numpy(dtype=float), matrix[coordinate_row, coordinate_gene]
                    )
                for metric, value in metrics.items():
                    donor_records.append({
                        "donor": donor,
                        "condition": conditions[rows[0]],
                        "fold": fold,
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "n_cells": int(len(rows)),
                    })

    if heldout_donors != set(np.unique(donors)):
        raise ValueError("cross-fitting did not hold out every donor exactly once")
    if not all(item["passed"] for item in leakage_checks):
        raise ValueError("one or more method contracts failed the leakage audit")
    if any(np.isnan(oof[method]).any() for method in learned_subdirs):
        raise ValueError("out-of-fold predictions are incomplete")
    print(f"phase=oof_done elapsed={_time.time():.0f}", flush=True)

    unit_metrics = pd.DataFrame(donor_records)
    donor_frame = unit_metrics[["donor", "condition"]].drop_duplicates().sort_values("donor")
    bootstrap_indices = stratified_bootstrap_indices(donor_frame, args.bootstrap, args.seed)
    unit_summary, unit_comparisons = summarize_unit_metrics(unit_metrics, bootstrap_indices)
    print(f"phase=unit_bootstrap_done elapsed={_time.time():.0f}", flush=True)
    disease_summary, disease_comparisons, disease_markers = summarize_disease_contrasts(
        oof, truth, donors, conditions, labels, feature_names, args.bootstrap, args.seed + 1
    )
    print(f"phase=disease_bootstrap_done elapsed={_time.time():.0f}", flush=True)
    summary = pd.concat([unit_summary, disease_summary], ignore_index=True)
    comparisons = pd.concat([unit_comparisons, disease_comparisons], ignore_index=True)

    print(f"phase=writing elapsed={_time.time():.0f}", flush=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    unit_metrics.to_parquet(output / "donor_metrics.parquet", index=False)
    summary.to_parquet(output / "bootstrap_summary.parquet", index=False)
    comparisons.to_parquet(output / "paired_comparisons.parquet", index=False)
    disease_markers.to_parquet(output / "disease_marker_effects.parquet", index=False)
    pd.DataFrame(leakage_checks).to_parquet(output / "leakage_checks.parquet", index=False)
    for method in learned_subdirs:
        np.save(output / f"oof_{method}.npy", oof[method], allow_pickle=False)
    report = {
        "design": "three-fold condition-stratified donor cross-fitting",
        "biological_unit": "donor",
        "donors": int(len(np.unique(donors))),
        "condition_counts": donor_frame["condition"].value_counts().sort_index().to_dict(),
        "bootstrap_replicates": args.bootstrap,
        "methods": sorted(unit_metrics["method"].unique()),
        "leakage_checks_passed": True,
        "artifacts": sorted(path.name for path in output.iterdir()),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
