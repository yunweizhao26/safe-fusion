from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from .io import dense
from .downstream import (
    adjusted_rand_index,
    infer_grn_scores,
    kmeans,
    macro_f1_nearest_centroid,
    marker_recovery,
    neighborhood_purity,
    normalized_mutual_information,
    pseudobulk_logfc,
    randomized_pca_embedding,
)
from .metrics import (
    average_precision_tie_aware,
    expected_budget_metrics,
    gaussian_crps,
    gaussian_nll,
    interval_coverage,
    log1p_mae,
    poisson_deviance,
    risk_coverage_auc,
    roc_auc_tie_aware,
    spearman,
)


def evaluate_core(
    adata: ad.AnnData,
    coordinates: pd.DataFrame,
    contract_directory: str | Path,
    corrupted_adata: ad.AnnData,
    method: str,
    dataset: str,
    corruption: str,
    fill_budgets: list[float],
) -> pd.DataFrame:
    validation_coordinates = coordinates.loc[coordinates["split"].eq("validation")].reset_index(drop=True)
    coordinates = coordinates.loc[coordinates["split"].eq("test")].reset_index(drop=True)
    if coordinates.empty:
        raise ValueError("locked test split has no evaluation coordinates")
    prediction = np.load(Path(contract_directory) / "mean.npy", allow_pickle=False)
    rows = coordinates["cell_index"].to_numpy(dtype=int)
    cols = coordinates["gene_index"].to_numpy(dtype=int)
    truth = coordinates["original_value"].to_numpy(dtype=float)
    predicted = prediction[rows, cols]
    validation_rows = validation_coordinates["cell_index"].to_numpy(dtype=int)
    validation_cols = validation_coordinates["gene_index"].to_numpy(dtype=int)
    validation_truth = validation_coordinates["original_value"].to_numpy(dtype=float)
    validation_predicted = prediction[validation_rows, validation_cols]
    score_path = Path(contract_directory) / "fill_score.npy"
    scores = np.load(score_path, allow_pickle=False)[rows, cols] if score_path.exists() else predicted
    labels = truth > coordinates["corrupted_value"].to_numpy(dtype=float)
    metrics = {
        "log1p_mae": log1p_mae(truth, predicted),
        "poisson_deviance": poisson_deviance(truth, predicted),
        "entry_spearman": spearman(truth, predicted),
    }
    if "clean_truth" in adata.layers:
        clean = dense(adata.layers["clean_truth"])
        corrupted = dense(corrupted_adata.layers["corrupted_counts"])
        test_cells = np.asarray(adata.obs_names.astype(str).isin(set(coordinates["cell_id"])))
        candidate = (corrupted == 0) & test_cells[:, None]
        zero_labels = clean[candidate] > 0
        zero_scores = np.load(score_path, allow_pickle=False)[candidate] if score_path.exists() else prediction[candidate]
        metrics.update({
            "roc_auc": roc_auc_tie_aware(zero_labels, zero_scores),
            "pr_auc": average_precision_tie_aware(zero_labels, zero_scores),
            "prevalence": float(zero_labels.mean()),
        })
    else:
        zero_labels, zero_scores = labels, scores
        # Controlled real-data corruptions provide known positives but no
        # defensible biological-zero negatives. Do not report a one-class PR
        # value of 1.0 as zero-selection evidence.
        if np.unique(labels).size < 2:
            metrics.update({"roc_auc": np.nan, "pr_auc": np.nan, "prevalence": float(labels.mean())})
        else:
            metrics.update({"roc_auc": roc_auc_tie_aware(labels, scores), "pr_auc": average_precision_tie_aware(labels, scores), "prevalence": float(labels.mean())})
    variance_path = Path(contract_directory) / "variance.npy"
    if variance_path.exists():
        variance = np.load(variance_path, allow_pickle=False)[rows, cols]
        metrics["interval_coverage_95"] = interval_coverage(truth, predicted, variance)
        metrics["gaussian_nll"] = gaussian_nll(truth, predicted, variance)
        metrics["gaussian_crps"] = gaussian_crps(truth, predicted, variance)
        metrics["risk_coverage_auc"] = risk_coverage_auc(truth, predicted, -variance)
    gene_correlations = []
    cell_correlations = []
    for _, index in coordinates.groupby("gene_id").groups.items():
        idx = np.asarray(list(index), dtype=int)
        if len(idx) > 1:
            gene_correlations.append(spearman(truth[idx], predicted[idx]))
    for _, index in coordinates.groupby("cell_id").groups.items():
        idx = np.asarray(list(index), dtype=int)
        if len(idx) > 1:
            cell_correlations.append(spearman(truth[idx], predicted[idx]))
    metrics["gene_wise_spearman"] = float(np.mean(gene_correlations)) if gene_correlations else float("nan")
    metrics["cell_wise_spearman"] = float(np.mean(cell_correlations)) if cell_correlations else float("nan")
    records = []
    for metric, value in metrics.items():
        records.append({"dataset": dataset, "corruption": corruption, "method": method, "scope": "overall", "biological_unit": "all", "metric": metric, "value": value, "fill_budget": np.nan})
    for budget in fill_budgets:
        budget_metrics = (
            expected_budget_metrics(zero_labels, zero_scores, budget)
            if np.unique(zero_labels).size >= 2
            else {"precision": np.nan, "recall": np.nan, "f1": np.nan, "false_fill_rate": np.nan}
        )
        for metric, value in budget_metrics.items():
            records.append({"dataset": dataset, "corruption": corruption, "method": method, "scope": "overall", "biological_unit": "all", "metric": metric, "value": value, "fill_budget": budget})
    for split_name, split_coordinates in (("validation", validation_coordinates), ("test", coordinates)):
        split_truth = validation_truth if split_name == "validation" else truth
        split_predicted = validation_predicted if split_name == "validation" else predicted
        for unit, indices in split_coordinates.groupby("biological_unit").groups.items():
            index = np.asarray(list(indices), dtype=int)
            records.append({
                "dataset": dataset, "corruption": corruption, "method": method,
                "scope": "biological_unit", "biological_unit": unit,
                "metric": "log1p_mae", "value": log1p_mae(split_truth[index], split_predicted[index]),
                "fill_budget": np.nan, "split": split_name, "n": int(len(index)),
            })
    return pd.DataFrame(records)


def evaluate_downstream(
    adata: ad.AnnData,
    split_frame: pd.DataFrame,
    contract_directory: str | Path,
    method: str,
    dataset: str,
    corruption: str,
    biological_unit: str,
    seed: int,
) -> pd.DataFrame:
    prediction = np.load(Path(contract_directory) / "mean.npy", allow_pickle=False)
    raw = dense(adata.layers["counts"])
    reference = dense(adata.layers["clean_truth"]) if "clean_truth" in adata.layers else raw
    raw_lib = raw.sum(axis=1)
    pred_lib = prediction.sum(axis=1)
    split = split_frame.set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    train, test = split != "test", split == "test"
    records = [{"dataset": dataset, "corruption": corruption, "method": method, "task": "library_size_preservation", "metric": "spearman", "value": spearman(raw_lib[test], pred_lib[test]), "status": "descriptive"}]
    if "clean_truth" in adata.layers:
        records.append({"dataset": dataset, "corruption": corruption, "method": method, "task": "simulation_clean_recovery", "metric": "log1p_mae", "value": log1p_mae(reference[test], prediction[test]), "status": "truth_bearing"})
    embedding = randomized_pca_embedding(prediction, train, seed)
    if "cell_type" in adata.obs:
        labels = adata.obs["cell_type"].astype(str).to_numpy()
        n_clusters = len(np.unique(labels[test]))
        clusterings = [kmeans(embedding[test], n_clusters, seed + offset) for offset in range(5)]
        records.extend([
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "clustering", "metric": "ari", "value": adjusted_rand_index(labels[test], clusterings[0]), "status": "evaluated"},
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "clustering", "metric": "nmi", "value": normalized_mutual_information(labels[test], clusterings[0]), "status": "evaluated"},
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "clustering", "metric": "neighborhood_purity", "value": neighborhood_purity(embedding[test], labels[test]), "status": "evaluated"},
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "clustering", "metric": "seed_stability_ari", "value": float(np.mean([adjusted_rand_index(clusterings[0], other) for other in clusterings[1:]])), "status": "evaluated"},
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "cell_identity", "metric": "donor_heldout_macro_f1", "value": macro_f1_nearest_centroid(embedding, labels, train, test), "status": "evaluated"},
        ])
        marker_ap, marker_rank = marker_recovery(prediction, reference, labels, train, test)
        records.extend([
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "marker_recovery", "metric": "pr_auc", "value": marker_ap, "status": "development_markers"},
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "marker_recovery", "metric": "rank_spearman", "value": marker_rank, "status": "development_markers"},
        ])
    if "condition" in adata.obs and biological_unit in adata.obs:
        try:
            test_obs = adata.obs.loc[test].copy()
            true_lfc = pseudobulk_logfc(reference[test], test_obs, biological_unit, "condition")
            pred_lfc = pseudobulk_logfc(prediction[test], test_obs, biological_unit, "condition")
            records.extend([
                {"dataset": dataset, "corruption": corruption, "method": method, "task": "differential_expression", "metric": "logfc_rmse", "value": float(np.sqrt(np.mean((true_lfc - pred_lfc) ** 2))), "status": "pseudobulk"},
                {"dataset": dataset, "corruption": corruption, "method": method, "task": "differential_expression", "metric": "logfc_spearman", "value": spearman(true_lfc, pred_lfc), "status": "pseudobulk"},
                {"dataset": dataset, "corruption": corruption, "method": method, "task": "differential_expression", "metric": "sign_agreement", "value": float(np.mean(np.sign(true_lfc) == np.sign(pred_lfc))), "status": "pseudobulk"},
            ])
        except ValueError as exc:
            records.append({"dataset": dataset, "corruption": corruption, "method": method, "task": "differential_expression", "metric": "not_evaluated", "value": np.nan, "status": str(exc)})
    if "pseudotime" in adata.obs:
        target = adata.obs["pseudotime"].to_numpy(dtype=float)
        design = np.column_stack([np.ones(train.sum()), embedding[train]])
        coefficients = np.linalg.lstsq(design, target[train], rcond=None)[0]
        estimate = np.column_stack([np.ones(test.sum()), embedding[test]]) @ coefficients
        records.append({"dataset": dataset, "corruption": corruption, "method": method, "task": "trajectory", "metric": "pseudotime_spearman", "value": spearman(target[test], estimate), "status": "heldout_unit"})
    if "lineage" in adata.obs:
        labels = adata.obs["lineage"].astype(str).to_numpy()
        records.append({"dataset": dataset, "corruption": corruption, "method": method, "task": "trajectory", "metric": "branch_macro_f1", "value": macro_f1_nearest_centroid(embedding, labels, train, test), "status": "heldout_unit"})
    edges = adata.uns.get("grn_edges", [])
    if len(edges) > 0:
        edge_labels, edge_scores = infer_grn_scores(prediction[test], adata.var_names.astype(str).tolist(), edges)
        prevalence = float(edge_labels.mean())
        auprc = average_precision_tie_aware(edge_labels, edge_scores)
        k = max(1, int(edge_labels.sum()))
        top = np.argsort(-edge_scores)[:k]
        records.extend([
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "grn", "metric": "edge_auprc", "value": auprc, "status": "known_simulation_edges"},
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "grn", "metric": "prevalence_lift", "value": auprc / prevalence if prevalence else np.nan, "status": "known_simulation_edges"},
            {"dataset": dataset, "corruption": corruption, "method": method, "task": "grn", "metric": "topk_jaccard", "value": float(edge_labels[top].sum() / (2 * k - edge_labels[top].sum())), "status": "known_simulation_edges"},
        ])
    return pd.DataFrame(records)
