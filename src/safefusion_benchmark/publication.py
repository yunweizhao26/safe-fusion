from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import cluster_bootstrap_difference


def weighted_group_means(frame: pd.DataFrame, value_column: str = "value", weight_column: str = "n") -> pd.Series:
    if weight_column in frame and (frame[weight_column].fillna(0) > 0).all():
        weighted = frame.assign(_weighted_value=frame[value_column] * frame[weight_column])
        sums = weighted.groupby("method")["_weighted_value"].sum()
        weights = frame.groupby("method")[weight_column].sum()
        return (sums / weights).astype(float)
    return frame.groupby("method")[value_column].mean()


def evaluate_publication_gate(metrics: pd.DataFrame, config: dict) -> dict:
    eligible = {
        name for name, spec in config["datasets"].items()
        if spec.get("enabled", True)
        and spec.get("accession") != "synthetic"
        and "quarantined" not in spec.get("role", "")
        and spec.get("role") != "trajectory_stage_only"
    }
    if not eligible:
        return {
            "claim": "not_evaluated",
            "reason": "synthetic smoke runs validate infrastructure but cannot satisfy publication gates",
            "criteria": {"eligible_real_datasets": 0, "continuous_passes": 0},
        }
    core = metrics[(metrics["analysis_family"] == "core") & metrics["dataset"].isin(eligible)]
    primary = core[(core["metric"] == "log1p_mae") & (core["scope"] == "biological_unit")]
    if "split" not in primary:
        return {
            "claim": "not_evaluated",
            "reason": "core per-unit metrics must carry a split column so the baseline is selected on validation units only",
            "criteria": {"eligible_real_datasets": len(eligible), "continuous_passes": 0},
        }
    required = float(config["study"]["minimum_relative_improvement"])
    replicates = int(config["study"]["bootstrap_replicates"])
    seed = int(config["study"]["seed"])
    evidence = []
    datasets_passed: set[str] = set()
    for (dataset, corruption), dataset_frame in primary.groupby(["dataset", "corruption"]):
        validation_frame = dataset_frame[dataset_frame["split"].eq("validation")]
        test_frame = dataset_frame[dataset_frame["split"].eq("test")]
        candidates = [method for method in dataset_frame["method"].unique() if method != "safe_fusion"]
        if "safe_fusion" not in set(test_frame["method"]) or not candidates or validation_frame.empty or test_frame.empty:
            continue
        validation_selection = validation_frame[validation_frame["method"].isin(candidates)]
        means = weighted_group_means(validation_selection)
        baseline = str(means.idxmin())
        comparison = cluster_bootstrap_difference(test_frame, "safe_fusion", baseline, "value", "biological_unit", replicates, seed, weight_column="n")
        test_selection = test_frame[test_frame["method"].isin([baseline, "safe_fusion"])]
        test_means = weighted_group_means(test_selection)
        baseline_mean = float(test_means[baseline])
        fusion_mean = float(test_means["safe_fusion"])
        improvement = (baseline_mean - fusion_mean) / baseline_mean if baseline_mean else float("nan")
        passed = bool(comparison["n_units"] >= 2 and improvement >= required and comparison["ci_high"] < 0)
        if passed:
            datasets_passed.add(dataset)
        evidence.append({"dataset": dataset, "corruption": corruption, "baseline": baseline, "relative_improvement": improvement, "paired_difference_fusion_minus_baseline": comparison, "passed": passed})
    continuous_passes = len(datasets_passed)
    enough_continuous = continuous_passes >= 2
    downstream = metrics[(metrics["analysis_family"] == "downstream") & metrics["dataset"].isin(eligible)]
    downstream_families = (
        int(downstream.loc[(downstream["method"] == "safe_fusion") & downstream["status"].isin(["evaluated", "pseudobulk", "heldout_unit", "known_simulation_edges"]), "task"].nunique())
        if not downstream.empty
        else 0
    )
    perturb = metrics[(metrics["analysis_family"] == "perturbation") & metrics["dataset"].isin(eligible)]
    perturb_roles_pass = set(perturb.loc[perturb["status"] == "passed", "perturbation_role"].dropna()) if "perturbation_role" in perturb and not perturb.empty else set()
    perturb_pass = {"ko", "crispra", "crispri"}.issubset(perturb_roles_pass)
    risk_pass = False
    all_pass = enough_continuous and downstream_families >= 2 and perturb_pass and risk_pass
    return {
        "claim": "best_imputation" if all_pass else "no_go",
        "reason": "all confirmatory gates passed" if all_pass else "one or more prespecified confirmatory gates failed or lack evidence",
        "criteria": {
            "eligible_real_datasets": len(eligible),
            "continuous_passes": continuous_passes,
            "continuous_required": 2,
            "downstream_task_families_with_evidence": downstream_families,
            "downstream_required": 2,
            "ko_crispra_crispri_pass": perturb_pass,
            "monotone_selective_risk_pass": risk_pass,
        },
        "continuous_evidence": evidence,
    }
