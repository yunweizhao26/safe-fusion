#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.mixture import GaussianMixture

from evaluate_papalexi_crossmodal import PROTEIN_TO_GENE, dense
from evaluate_pdl1_state_baselines import IFNG_RESPONSE_GENES, QUANTILES, VALUE_RANKINGS

PROTEIN_NAMES = {"PDL1": "PD-L1", "CD86": "CD86", "PDL2": "PD-L2", "CD366": "TIM-3"}
CONTROL = "NT"
SCALES = {"clr": "adt_clr", "count": "adt_count"}
RNA_RANKINGS = ("safe_fusion", "svd", "weighted_knn", "scvi", "fused_value")
RANKINGS = (*RNA_RANKINGS, "library_size", "target_baseline", "ifng_score", "cell_zero_score")
COVARIATE_SETS = {
    "target_ifng_library": ("target", "ifng_score", "library_size"),
    "baseline_ifng_library": ("target_baseline", "ifng_score", "library_size"),
    "target_ifng_library_replicate": ("target", "ifng_score", "library_size", "replicate"),
    "target_ifng_library_cellscore": ("target", "ifng_score", "library_size", "cell_zero_score"),
}
CONTROL_COVARIATE_SETS = {
    "ifng_library": ("ifng_score", "library_size"),
    "ifng_library_replicate": ("ifng_score", "library_size", "replicate"),
    "ifng_library_cellscore": ("ifng_score", "library_size", "cell_zero_score"),
}
CONTINUOUS_COVARIATES = ("target_baseline", "ifng_score", "library_size", "cell_zero_score")
MIN_GROUP_CELLS = 3
DETECTED_RNA_LEVELS = ("rna_count", "rna_log_cp10k")


def rank_correlation(ranks: np.ndarray, target_ranks: np.ndarray) -> np.ndarray:
    centered = ranks - ranks.mean(axis=0)
    target = target_ranks - target_ranks.mean()
    denominator = np.sqrt((centered**2).sum(axis=0) * (target**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, centered.T @ target / denominator, np.nan)


def mean_auroc(ranks: np.ndarray, protein: np.ndarray) -> np.ndarray:
    values = []
    for threshold in np.quantile(protein, QUANTILES):
        high = protein > threshold
        n_high = int(high.sum())
        n_low = len(high) - n_high
        if n_high and n_low:
            values.append((ranks[high].sum(axis=0) - n_high * (n_high + 1) / 2) / (n_high * n_low))
    return np.mean(values, axis=0)


def within_group_spearman(scores: np.ndarray, protein: np.ndarray, groups: np.ndarray) -> np.ndarray:
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[order])) + 1
    total = np.zeros(scores.shape[1])
    weight = np.zeros(scores.shape[1])
    for members in np.split(order, boundaries):
        if len(members) < MIN_GROUP_CELLS:
            continue
        rho = rank_correlation(rankdata(scores[members], axis=0), rankdata(protein[members]))
        defined = np.isfinite(rho)
        total[defined] += len(members) * rho[defined]
        weight[defined] += len(members)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(weight > 0, total / weight, np.nan)


def indicators(codes: np.ndarray) -> np.ndarray:
    return (codes[:, None] == np.unique(codes)[None, :]).astype(float)


def residuals(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def column_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=0)
    right = right - right.mean()
    denominator = np.sqrt((left**2).sum(axis=0) * (right**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, left.T @ right / denominator, np.nan)


@dataclass
class Job:
    protein: str
    gene: str
    scale: str
    cells: str
    scope: str
    resampling: str
    rankings: tuple[str, ...]
    scores: np.ndarray
    response: np.ndarray
    groups: np.ndarray
    replicate: np.ndarray
    covariates: dict[str, np.ndarray]
    within: bool
    covariate_sets: dict[str, tuple[str, ...]]
    seed: tuple[int, ...]
    draws: int
    group_names: list[str] = field(default_factory=list)


def statistics(job: Job, index: np.ndarray, groups: np.ndarray) -> dict[tuple[str, str, str], float]:
    scores = job.scores[index]
    response = job.response[index]
    ranks = rankdata(scores, axis=0)
    response_ranks = rankdata(response)
    values: dict[tuple[str, str, str], float] = {}
    pooled = rank_correlation(ranks, response_ranks)
    auroc = mean_auroc(ranks, response)
    for column, ranking in enumerate(job.rankings):
        values[("pooled_spearman", "", ranking)] = float(pooled[column])
        values[("pooled_mean_auroc", "", ranking)] = float(auroc[column])
    if job.within:
        within = within_group_spearman(scores, response, groups)
        for column, ranking in enumerate(job.rankings):
            if ranking != "target_baseline":
                values[("within_target_spearman", "", ranking)] = float(within[column])
    for set_name, covariates in job.covariate_sets.items():
        columns = [np.ones(len(index))]
        for covariate in covariates:
            if covariate == "target":
                columns.append(indicators(groups))
            elif covariate == "replicate":
                columns.append(indicators(job.replicate[index]))
            else:
                columns.append(rankdata(job.covariates[covariate][index])[:, None])
        design = np.column_stack(columns)
        response_residual = residuals(design, response_ranks)
        eligible = [
            column for column, ranking in enumerate(job.rankings)
            if ranking not in covariates and not (ranking == "target_baseline" and "target" in covariates)
        ]
        partial = column_correlation(residuals(design, ranks[:, eligible]), response_residual)
        for position, column in enumerate(eligible):
            values[("partial_spearman", set_name, job.rankings[column])] = float(partial[position])
        if {"safe_fusion", "svd"} <= set(job.rankings):
            safe_fusion = job.rankings.index("safe_fusion")
            svd = job.rankings.index("svd")
            for name, own, other in (("safe_fusion_given_svd", safe_fusion, svd), ("svd_given_safe_fusion", svd, safe_fusion)):
                extended = np.column_stack([design, ranks[:, other]])
                values[("partial_spearman_adding_other_ranking", set_name, name)] = float(
                    column_correlation(residuals(extended, ranks[:, [own]]), residuals(extended, response_ranks))[0]
                )
    return values


def run_job(job: Job) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_cells = len(job.response)
    point = statistics(job, np.arange(n_cells), job.groups)
    rng = np.random.default_rng(list(job.seed))
    clusters = [np.flatnonzero(job.groups == code) for code in np.unique(job.groups)]
    replicates = [np.flatnonzero(job.replicate == code) for code in np.unique(job.replicate)]
    samples = []
    for _ in range(job.draws):
        if job.resampling == "target_clusters":
            chosen = rng.choice(len(clusters), size=len(clusters), replace=True)
            index = np.concatenate([clusters[cluster] for cluster in chosen])
            groups = np.concatenate([np.full(len(clusters[cluster]), copy) for copy, cluster in enumerate(chosen)])
        else:
            index = np.concatenate([rng.choice(members, size=len(members), replace=True) for members in replicates])
            groups = job.groups[index]
        samples.append(statistics(job, index, groups))
    samples = pd.DataFrame(samples)
    base = {
        "protein": PROTEIN_NAMES[job.protein], "gene": job.gene, "scale": job.scale, "cells": job.cells,
        "scope": job.scope, "n_cells": n_cells, "n_groups": len(clusters), "resampling": job.resampling,
        "draws": job.draws,
    }
    rows, paired = [], []
    for key, estimate in point.items():
        statistic, covariates, ranking = key
        draws = samples[key].to_numpy(dtype=float)
        rows.append({
            **base, "statistic": statistic, "covariates": covariates, "ranking": ranking, "estimate": estimate,
            "ci_low": float(np.nanquantile(draws, 0.025)), "ci_high": float(np.nanquantile(draws, 0.975)),
            "defined_draws": int(np.isfinite(draws).sum()),
        })
        reference = (statistic, covariates, "safe_fusion")
        if ranking == "safe_fusion" or reference not in point:
            continue
        difference = samples[reference].to_numpy(dtype=float) - draws
        paired.append({
            **base, "statistic": statistic, "covariates": covariates, "comparator": ranking,
            "safe_fusion_minus_comparator": point[reference] - estimate,
            "ci_low": float(np.nanquantile(difference, 0.025)), "ci_high": float(np.nanquantile(difference, 0.975)),
        })
    return pd.DataFrame(rows), pd.DataFrame(paired)


def load_inputs(prepared: Path, benchmark: Path, panel_path: Path) -> dict:
    truth = ad.read_h5ad(prepared)
    corrupted = ad.read_h5ad(benchmark / "corrupted.h5ad")
    if not np.array_equal(truth.obs_names.astype(str), corrupted.obs_names.astype(str)):
        raise ValueError("Prepared and corrupted cell orders differ")
    if not np.array_equal(truth.var_names.astype(str), corrupted.var_names.astype(str)):
        raise ValueError("Prepared and corrupted gene orders differ")
    gene_index = {str(gene): index for index, gene in enumerate(truth.var_names)}
    counts = dense(corrupted.layers["corrupted_counts"])
    library = counts.sum(axis=1)
    normalized = np.log1p(counts / np.maximum(library, 1)[:, None] * 1e4)
    truth_counts = dense(truth.layers["counts"] if "counts" in truth.layers else truth.X)
    cells = pd.DataFrame({
        "cell_index": np.arange(truth.n_obs),
        "cell_id": truth.obs_names.astype(str),
        "source_cell_id": truth.obs["source_cell_id"].astype(str).to_numpy(),
        "library_size": np.log1p(library),
        "ifng_score": normalized[:, [gene_index[gene] for gene in IFNG_RESPONSE_GENES]].mean(axis=1),
    })
    cells["split"] = pd.read_parquet(benchmark / "splits.parquet").set_index("cell_id").loc[cells["cell_id"], "split"].to_numpy()
    scores = pd.read_parquet(benchmark / "mlp_selector" / "selected_gene_scores.parquet")
    scores = scores.loc[scores["split"].astype(str) == "test", ["cell_id", "gene_id", "selector_score", "masked_positive"]]
    return {
        "gene_index": gene_index,
        "counts": counts,
        "truth_counts": truth_counts,
        "cells": cells,
        "panel": pd.read_parquet(panel_path),
        "scores": scores,
        "values": {name: np.load(benchmark / contract / "mean.npy", mmap_mode="r") for name, contract in VALUE_RANKINGS.items()},
    }


def cell_zero_score(frame: pd.DataFrame, scores: pd.DataFrame, gene: str) -> np.ndarray:
    test = scores.loc[scores["split"].astype(str) == "test"]
    total = test.groupby("cell_index")["selector_score"].agg(["sum", "count"])
    own = test.loc[test["gene_id"].astype(str) == gene].set_index("cell_index")["selector_score"]
    cells = frame["cell_index"].to_numpy(dtype=int)
    own_score = own.reindex(cells).to_numpy(dtype=float)
    has_own = np.isfinite(own_score)
    other_sum = total["sum"].reindex(cells).to_numpy(dtype=float) - np.where(has_own, own_score, 0.0)
    other_count = total["count"].reindex(cells).to_numpy(dtype=float) - has_own
    return other_sum / other_count


def protein_table(inputs: dict, protein: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = inputs["panel"]
    panel = panel.loc[panel["protein"].astype(str) == protein, ["cell_id", "replicate", "target", *SCALES.values()]]
    panel = panel.rename(columns={"cell_id": "source_cell_id"})
    cells = inputs["cells"].merge(panel, on="source_cell_id", validate="one_to_one")
    baselines = cells.loc[cells["split"] == "development"].groupby("target")[list(SCALES.values())].mean()
    return cells, baselines


def recorded_zero_frame(inputs: dict, protein: str, gene: str) -> pd.DataFrame:
    cells, baselines = protein_table(inputs, protein)
    column = inputs["gene_index"][gene]
    frame = inputs["scores"].loc[inputs["scores"]["gene_id"].astype(str) == gene]
    frame = frame.merge(cells, on="cell_id", validate="one_to_one")
    rows = frame["cell_index"].to_numpy(dtype=int)
    recorded_zero = (inputs["truth_counts"][rows, column] == 0) & (frame["masked_positive"].astype(int).to_numpy() == 0)
    frame = frame.loc[recorded_zero].reset_index(drop=True)
    rows = frame["cell_index"].to_numpy(dtype=int)
    frame["safe_fusion"] = frame["selector_score"].astype(float)
    for name, matrix in inputs["values"].items():
        frame[name] = np.asarray(matrix[rows, column], dtype=float)
    for scale, source in SCALES.items():
        frame[f"target_baseline_{scale}"] = frame["target"].map(baselines[source]).astype(float)
    if frame.filter(like="target_baseline_").isna().any().any():
        raise ValueError(f"{protein}: a held-out perturbation target has no development cells")
    return frame


def detected_rna_frame(inputs: dict, protein: str, gene: str) -> pd.DataFrame:
    cells, baselines = protein_table(inputs, protein)
    truth = inputs["truth_counts"]
    column = inputs["gene_index"][gene]
    detected = truth[cells["cell_index"].to_numpy(), column] > 0
    frame = cells.loc[(cells["split"] == "test").to_numpy() & detected].reset_index(drop=True)
    rows = frame["cell_index"].to_numpy(dtype=int)
    frame["rna_count"] = truth[rows, column].astype(float)
    frame["rna_log_cp10k"] = np.log1p(truth[rows, column] / truth[rows].sum(axis=1) * 1e4)
    for scale, source in SCALES.items():
        frame[f"target_baseline_{scale}"] = frame["target"].map(baselines[source]).astype(float)
    return frame


def make_jobs(frames: dict, draws: int, seed: int) -> list[Job]:
    jobs = []
    for protein_number, (protein, gene) in enumerate(PROTEIN_TO_GENE.items()):
        for scale_number, (scale, source) in enumerate(SCALES.items()):
            for cells_number, cells in enumerate(("recorded_rna_zero", "detected_rna")):
                frame = frames[(protein, cells)]
                rankings = RANKINGS if cells == "recorded_rna_zero" else DETECTED_RNA_LEVELS
                scopes = [("all", frame, "target_clusters"), ("all", frame, "cells_within_replicate")]
                if cells == "recorded_rna_zero":
                    scopes.append(("control", frame.loc[frame["target"] == CONTROL], "cells_within_replicate"))
                    scopes.extend(
                        (f"replicate_{replicate}", group, "target_clusters")
                        for replicate, group in frame.groupby("replicate", observed=True)
                    )
                for scope_number, (scope, subset, resampling) in enumerate(scopes):
                    subset = subset.reset_index(drop=True)
                    scope_rankings = tuple(r for r in rankings if not (scope == "control" and r == "target_baseline"))
                    data = subset.assign(target_baseline=subset[f"target_baseline_{scale}"])
                    group_names, groups = np.unique(data["target"].astype(str), return_inverse=True)
                    by_targets = resampling == "target_clusters"
                    if scope == "control":
                        covariate_sets = CONTROL_COVARIATE_SETS
                    elif scope == "all" and by_targets:
                        covariate_sets = COVARIATE_SETS
                    else:
                        covariate_sets = {}
                    jobs.append(Job(
                        protein=protein, gene=gene, scale=scale, cells=cells, scope=scope, resampling=resampling,
                        rankings=scope_rankings,
                        scores=data[list(scope_rankings)].to_numpy(dtype=float),
                        response=data[source].to_numpy(dtype=float),
                        groups=groups,
                        replicate=np.unique(data["replicate"].astype(str), return_inverse=True)[1],
                        covariates={name: data[name].to_numpy(dtype=float) for name in CONTINUOUS_COVARIATES},
                        within=by_targets,
                        covariate_sets=covariate_sets,
                        seed=(seed, protein_number, scale_number, cells_number, scope_number),
                        draws=draws,
                        group_names=list(group_names),
                    ))
    return jobs


SPECIFICITY_RANKINGS = (*RNA_RANKINGS, "library_size", "ifng_score")


@dataclass
class SpecificityJob:
    gene: str
    protein: str
    scale: str
    scores: np.ndarray
    responses: np.ndarray
    groups: np.ndarray
    seed: tuple[int, ...]
    draws: int


def specificity_statistics(job: SpecificityJob, index: np.ndarray, groups: np.ndarray) -> dict[tuple[str, str, str], float]:
    scores = job.scores[index]
    ranks = rankdata(scores, axis=0)
    values = {}
    for column, protein in enumerate(PROTEIN_TO_GENE):
        response = job.responses[index, column]
        pooled = rank_correlation(ranks, rankdata(response))
        within = within_group_spearman(scores, response, groups)
        for position, ranking in enumerate(SPECIFICITY_RANKINGS):
            values[("pooled_spearman", ranking, protein)] = float(pooled[position])
            values[("within_target_spearman", ranking, protein)] = float(within[position])
    return values


def run_specificity(job: SpecificityJob) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = specificity_statistics(job, np.arange(len(job.groups)), job.groups)
    rng = np.random.default_rng(list(job.seed))
    clusters = [np.flatnonzero(job.groups == code) for code in np.unique(job.groups)]
    samples = []
    for _ in range(job.draws):
        chosen = rng.choice(len(clusters), size=len(clusters), replace=True)
        index = np.concatenate([clusters[cluster] for cluster in chosen])
        groups = np.concatenate([np.full(len(clusters[cluster]), copy) for copy, cluster in enumerate(chosen)])
        samples.append(specificity_statistics(job, index, groups))
    samples = pd.DataFrame(samples)
    base = {
        "gene": job.gene, "encoded_protein": PROTEIN_NAMES[job.protein], "scale": job.scale,
        "n_cells": len(job.groups), "n_groups": len(clusters), "resampling": "target_clusters", "draws": job.draws,
    }
    rows, paired = [], []
    for key, estimate in point.items():
        statistic, ranking, protein = key
        draws = samples[key].to_numpy(dtype=float)
        rows.append({
            **base, "statistic": statistic, "ranking": ranking, "protein": PROTEIN_NAMES[protein],
            "encoded": protein == job.protein, "estimate": estimate,
            "ci_low": float(np.nanquantile(draws, 0.025)), "ci_high": float(np.nanquantile(draws, 0.975)),
        })
        if protein == job.protein:
            continue
        encoded = (statistic, ranking, job.protein)
        difference = samples[encoded].to_numpy(dtype=float) - draws
        paired.append({
            **base, "statistic": statistic, "ranking": ranking, "other_protein": PROTEIN_NAMES[protein],
            "encoded_minus_other": point[encoded] - estimate,
            "ci_low": float(np.nanquantile(difference, 0.025)), "ci_high": float(np.nanquantile(difference, 0.975)),
        })
    return pd.DataFrame(rows), pd.DataFrame(paired)


def make_specificity_jobs(inputs: dict, frames: dict, draws: int, seed: int) -> list[SpecificityJob]:
    panel = inputs["panel"]
    jobs = []
    for protein_number, (protein, gene) in enumerate(PROTEIN_TO_GENE.items()):
        frame = frames[(protein, "recorded_rna_zero")]
        groups = np.unique(frame["target"].astype(str), return_inverse=True)[1]
        for scale_number, (scale, source) in enumerate(SCALES.items()):
            wide = panel.pivot(index="cell_id", columns="protein", values=source)
            responses = wide.loc[frame["source_cell_id"], list(PROTEIN_TO_GENE)].to_numpy(dtype=float)
            jobs.append(SpecificityJob(
                gene=gene, protein=protein, scale=scale,
                scores=frame[list(SPECIFICITY_RANKINGS)].to_numpy(dtype=float),
                responses=responses, groups=groups,
                seed=(seed, 100 + protein_number, scale_number), draws=draws,
            ))
    return jobs


def by_target_table(frames: dict) -> pd.DataFrame:
    rows = []
    for protein, gene in PROTEIN_TO_GENE.items():
        frame = frames[(protein, "recorded_rna_zero")]
        for scale, source in SCALES.items():
            for target, group in frame.groupby("target", observed=True):
                row = {"protein": PROTEIN_NAMES[protein], "gene": gene, "scale": scale, "target": target, "n_cells": len(group)}
                for ranking in (*RNA_RANKINGS, "library_size", "ifng_score", "cell_zero_score"):
                    rho = rank_correlation(rankdata(group[[ranking]].to_numpy(), axis=0), rankdata(group[source].to_numpy()))[0]
                    row[ranking] = float(rho) if len(group) >= MIN_GROUP_CELLS else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    counts = np.bincount(groups)
    means = np.bincount(groups, weights=values) / np.maximum(counts, 1)
    total = ((values - values.mean()) ** 2).sum()
    return float((counts * (means - values.mean()) ** 2).sum() / total)


def variance_explained(inputs: dict, frames: dict, draws: int, seed: int) -> pd.DataFrame:
    rows = []
    panel = inputs["panel"]
    for protein_number, (protein, gene) in enumerate(PROTEIN_TO_GENE.items()):
        matched = panel.loc[panel["protein"].astype(str) == protein]
        sets = {"all_matched_cells": matched, "recorded_rna_zero_test_cells": frames[(protein, "recorded_rna_zero")]}
        for set_number, (name, table) in enumerate(sets.items()):
            groups = np.unique(table["target"].astype(str), return_inverse=True)[1]
            members = [np.flatnonzero(groups == code) for code in np.unique(groups)]
            for scale_number, (scale, values) in enumerate({
                "clr": table["adt_clr"].to_numpy(dtype=float),
                "log_count": np.log1p(table["adt_count"].to_numpy(dtype=float)),
            }.items()):
                rng = np.random.default_rng([seed, protein_number, set_number, scale_number])
                samples = []
                for _ in range(draws):
                    index = np.concatenate([rng.choice(cells, size=len(cells), replace=True) for cells in members])
                    samples.append(eta_squared(values[index], groups[index]))
                rows.append({
                    "protein": PROTEIN_NAMES[protein], "gene": gene, "cells": name, "scale": scale,
                    "n_cells": len(values), "n_groups": len(members), "eta_squared": eta_squared(values, groups),
                    "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975)),
                    "resampling": "cells_within_target", "draws": draws,
                })
    return pd.DataFrame(rows)


def background_summary(inputs: dict, frames: dict, seed: int) -> pd.DataFrame:
    rows = []
    panel = inputs["panel"]
    for protein, gene in PROTEIN_TO_GENE.items():
        values = np.log1p(panel.loc[panel["protein"].astype(str) == protein, "adt_count"].to_numpy(dtype=float))
        mixture = GaussianMixture(n_components=2, random_state=seed).fit(values[:, None])
        background = int(np.argmin(mixture.means_.ravel()))
        mean = float(mixture.means_.ravel()[background])
        sd = float(np.sqrt(mixture.covariances_.ravel()[background]))
        zeros = np.log1p(frames[(protein, "recorded_rna_zero")]["adt_count"].to_numpy(dtype=float))
        above = lambda x: float((mixture.predict_proba(x[:, None])[:, background] < 0.5).mean())
        rows.append({
            "protein": PROTEIN_NAMES[protein], "gene": gene, "n_matched_cells": len(values),
            "background_log_count_mean": mean, "background_log_count_sd": sd,
            "background_weight": float(mixture.weights_[background]),
            "signal_log_count_mean": float(mixture.means_.ravel()[1 - background]),
            "fraction_above_background_all_matched": above(values),
            "fraction_above_background_recorded_rna_zero_test": above(zeros),
            "median_background_normalized_recorded_rna_zero_test": float(np.median((zeros - mean) / sd)),
        })
    return pd.DataFrame(rows)


def target_table(inputs: dict, frames: dict) -> pd.DataFrame:
    cells, baselines = protein_table(inputs, "PDL1")
    column = inputs["gene_index"]["CD274"]
    rows = cells["cell_index"].to_numpy()
    cells["cd274_detected"] = inputs["truth_counts"][rows, column] > 0
    cells["cd274_masked"] = cells["cd274_detected"] & (inputs["counts"][rows, column] == 0)
    zeros = frames[("PDL1", "recorded_rna_zero")]
    test = cells.loc[cells["split"] == "test"]
    table = pd.DataFrame({
        "development_cells": cells.loc[cells["split"] == "development"].groupby("target").size(),
        "test_cells": test.groupby("target").size(),
        "test_cells_with_detected_cd274": test.groupby("target")["cd274_detected"].sum(),
        "test_cells_with_masked_cd274": test.groupby("target")["cd274_masked"].sum(),
        "test_cells_with_recorded_cd274_zero": zeros.groupby("target").size(),
        "target_baseline_pdl1_clr": baselines["adt_clr"],
        "mean_pdl1_clr_recorded_cd274_zero": zeros.groupby("target")["adt_clr"].mean(),
    })
    table.index.name = "target"
    table = table.fillna({"test_cells_with_recorded_cd274_zero": 0}).reset_index()
    counts = ["development_cells", "test_cells", "test_cells_with_detected_cd274", "test_cells_with_masked_cd274", "test_cells_with_recorded_cd274_zero"]
    table[counts] = table[counts].astype(int)
    table["control"] = table["target"] == CONTROL
    return table.sort_values(["control", "target"]).reset_index(drop=True)


def global_fill(inputs: dict, frames: dict, rerun: pd.DataFrame, selector_dir: Path, benchmark: Path) -> tuple[pd.DataFrame, dict]:
    report = json.loads((selector_dir / "calibration_report.json").read_text())
    fused = np.load(benchmark / "safe_fusion" / "mean.npy", mmap_mode="r")
    production = pd.read_parquet(benchmark / "mlp_selector" / "selected_gene_scores.parquet")
    keys = ["split", "cell_id", "gene_id"]
    joined = production[[*keys, "selector_score"]].merge(rerun[[*keys, "selector_score"]], on=keys, validate="one_to_one")
    difference = float(np.abs(joined["selector_score_x"] - joined["selector_score_y"]).max())
    rows = []
    for directory in sorted(selector_dir.glob("safe_fusion_calibrated_mlp_topk_*")):
        budget = float(json.loads((directory / "metadata.json").read_text())["parameters"]["decision_layer"]["fill_budget"])
        output = np.load(directory / "mean.npy", mmap_mode="r")
        for protein, gene in PROTEIN_TO_GENE.items():
            frame = frames[(protein, "recorded_rna_zero")]
            rows_index = frame["cell_index"].to_numpy(dtype=int)
            column = inputs["gene_index"][gene]
            masked = rerun.loc[(rerun["split"] == "test") & (rerun["gene_id"] == gene) & (rerun["masked_positive"] == 1), "cell_index"].to_numpy(dtype=int)
            if np.any(np.asarray(fused[np.concatenate([rows_index, masked]), column]) == 0):
                raise ValueError(f"{gene}: a zero fused value would make a fill invisible")
            filled = np.asarray(output[rows_index, column]) != 0
            clr = frame["adt_clr"].to_numpy(dtype=float)
            zero_scores = rerun.loc[(rerun["split"] == "test") & (rerun["gene_id"] == gene)].set_index("cell_id").loc[frame["cell_id"], "selector_score"]
            rows.append({
                "global_fill_fraction": budget, "protein": PROTEIN_NAMES[protein], "gene": gene,
                "test_candidate_zeros": int(report["test"]["n_zeros"]),
                "selector_threshold": float(report["budgets"][str(budget)]["threshold"]),
                "max_selector_score_recorded_zeros": float(zero_scores.max()),
                "masked_test_entries": len(masked), "filled_masked_test_entries": int((np.asarray(output[masked, column]) != 0).sum()),
                "recorded_zeros": len(frame), "filled_recorded_zeros": int(filled.sum()),
                "filled_share": float(filled.mean()),
                "filled_share_over_global_fraction": float(filled.mean() / budget),
                "mean_protein_clr_filled": float(clr[filled].mean()) if filled.any() else np.nan,
                "mean_protein_clr_not_filled": float(clr[~filled].mean()) if (~filled).any() else np.nan,
            })
    return pd.DataFrame(rows), {"max_abs_score_difference_from_production_selector": difference, "rows_compared": len(joined)}


def pdl1_pooled_table(association: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = association.loc[
        (association["protein"] == "PD-L1") & (association["scale"] == "clr") & (association["cells"] == "recorded_rna_zero")
        & (association["scope"] == "all") & association["statistic"].isin(["pooled_spearman", "pooled_mean_auroc"])
    ]
    table = rows.pivot_table(index="ranking", columns=["statistic", "resampling"], values=["estimate", "ci_low", "ci_high"], sort=False)
    out = pd.DataFrame(index=pd.Index(RANKINGS, name="ranking"))
    for statistic, label in (("pooled_spearman", "spearman"), ("pooled_mean_auroc", "mean_auroc")):
        out[label] = table[("estimate", statistic, "target_clusters")]
        for resampling, suffix in (("target_clusters", "target"), ("cells_within_replicate", "cell")):
            out[f"{label}_ci_low_{suffix}"] = table[("ci_low", statistic, resampling)]
            out[f"{label}_ci_high_{suffix}"] = table[("ci_high", statistic, resampling)]
        difference = paired.loc[
            (paired["protein"] == "PD-L1") & (paired["scale"] == "clr") & (paired["cells"] == "recorded_rna_zero")
            & (paired["scope"] == "all") & (paired["statistic"] == statistic) & (paired["resampling"] == "target_clusters")
        ].set_index("comparator")
        out[f"safe_fusion_minus_{label}"] = difference["safe_fusion_minus_comparator"]
        out[f"safe_fusion_minus_{label}_ci_low_target"] = difference["ci_low"]
        out[f"safe_fusion_minus_{label}_ci_high_target"] = difference["ci_high"]
    out.insert(0, "n_cells", int(rows["n_cells"].iloc[0]))
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=Path("external_data/prepared/papalexi_eccite_crossmodal.h5ad"))
    parser.add_argument("--benchmark-root", type=Path, default=Path("artifacts/paper_evidence/papalexi_crossmodal/benchmark"))
    parser.add_argument("--panel", type=Path, default=Path("artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet"))
    parser.add_argument("--global-fill-dir", type=Path, default=Path("artifacts/paper_evidence/review_round2/protein/global_fill_selector"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/paper_evidence/review_round2/protein/evaluation"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()

    inputs = load_inputs(args.prepared, args.benchmark_root, args.panel)
    all_gene_scores = pd.read_parquet(
        args.global_fill_dir / "selected_gene_scores.parquet",
        columns=["split", "cell_index", "cell_id", "gene_id", "selector_score", "masked_positive"],
    )
    frames = {}
    for protein, gene in PROTEIN_TO_GENE.items():
        frames[(protein, "recorded_rna_zero")] = recorded_zero_frame(inputs, protein, gene)
        frames[(protein, "detected_rna")] = detected_rna_frame(inputs, protein, gene)
        for cells in ("recorded_rna_zero", "detected_rna"):
            frame = frames[(protein, cells)]
            frame["cell_zero_score"] = cell_zero_score(frame, all_gene_scores, gene)
    jobs = make_jobs(frames, args.bootstrap, args.seed)
    specificity_jobs = make_specificity_jobs(inputs, frames, args.bootstrap, args.seed)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_job, jobs))
        specificity = list(pool.map(run_specificity, specificity_jobs))
    association = pd.concat([result[0] for result in results], ignore_index=True)
    paired = pd.concat([result[1] for result in results], ignore_index=True)
    fill, fill_check = global_fill(inputs, frames, all_gene_scores, args.global_fill_dir, args.benchmark_root)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    association.to_csv(output / "association.csv", index=False)
    paired.to_csv(output / "paired_differences.csv", index=False)
    pdl1_pooled_table(association, paired).to_csv(output / "pdl1_pooled_rankings.csv", index=False)
    pd.concat([result[0] for result in specificity], ignore_index=True).to_csv(output / "protein_specificity.csv", index=False)
    pd.concat([result[1] for result in specificity], ignore_index=True).to_csv(output / "protein_specificity_paired.csv", index=False)
    by_target_table(frames).to_csv(output / "within_target_by_target.csv", index=False)
    variance_explained(inputs, frames, args.bootstrap, args.seed).to_csv(output / "variance_explained.csv", index=False)
    background_summary(inputs, frames, args.seed).to_csv(output / "protein_background.csv", index=False)
    targets = target_table(inputs, frames)
    targets.to_csv(output / "cd274_zero_targets.csv", index=False)
    fill.to_csv(output / "global_fill.csv", index=False)
    panel_targets = sorted(inputs["panel"]["target"].astype(str).unique())
    report = {
        "n_cells": {f"{PROTEIN_NAMES[p]}:{c}": int(len(frames[(p, c)])) for p, c in frames},
        "benchmark_cells_with_matched_protein": {
            split: int(n) for split, n in protein_table(inputs, "PDL1")[0].groupby("split").size().items()
        },
        "benchmark_cells": {split: int(n) for split, n in inputs["cells"].groupby("split").size().items()},
        "benchmark_targets": [t for t in targets["target"] if t != CONTROL],
        "matched_panel_targets": [t for t in panel_targets if t != CONTROL],
        "cd274_is_benchmark_target": "CD274" in set(targets["target"]),
        "cd274_is_matched_panel_target": "CD274" in set(panel_targets),
        "control_group": CONTROL,
        "scales": {"clr": "log1p count minus the mean log1p count of the four proteins in the cell", "count": "raw antibody count"},
        "background_normalization": (
            "log1p count standardized by the background component of a two-component Gaussian mixture per protein over "
            "all matched cells; an increasing map per protein, so rank statistics equal those of the raw count"
        ),
        "covariate_sets": {**COVARIATE_SETS, **CONTROL_COVARIATE_SETS},
        "cell_zero_score": "mean selector score of the cell's held-out candidate zeros of all genes other than the evaluated gene",
        "all_gene_test_candidates_scored": int((all_gene_scores["split"].astype(str) == "test").sum()),
        "partial_correlation": (
            "Pearson correlation of the residuals of the ranking ranks and protein ranks after least-squares regression "
            "on an intercept, target or replicate indicators and the ranks of continuous covariates"
        ),
        "within_target": f"Spearman inside each target group (non-targeting cells are one group), cell-weighted mean over groups with at least {MIN_GROUP_CELLS} cells",
        "resampling": {
            "target_clusters": "targets and the non-targeting group resampled with replacement; each drawn copy is its own group",
            "cells_within_replicate": "cells resampled with replacement within each replicate, as in Supplementary Table S9",
            "cells_within_target": "cells resampled with replacement within each target (variance explained)",
        },
        "bootstrap_draws": args.bootstrap,
        "seed": args.seed,
        "global_fill_selector": str(args.global_fill_dir),
        "global_fill_selector_check": fill_check,
    }
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
