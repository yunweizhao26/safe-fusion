#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

FILL_FRACTIONS = ("0p01", "0p05", "0p1")


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def prepare(args: argparse.Namespace) -> None:
    adata = ad.read_h5ad(args.input)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    labels = adata.obs[args.label_column].astype(str).to_numpy()
    control = labels == args.control_label
    rng = np.random.default_rng(args.seed)
    pseudo = labels.astype(object).copy()
    splits = sorted(set(split))
    sizes = {name: int(np.median(pd.Series(labels[(split == name) & ~control]).value_counts())) for name in splits}
    total_size = sum(sizes.values())
    n_groups = int(round(args.pseudo_share * control.sum() / total_size))
    if n_groups < 1:
        raise ValueError("too few control cells for one pseudo-perturbation group")
    groups = [f"pseudo_{index:02d}" for index in range(n_groups)]
    for name in splits:
        cells = rng.permutation(np.flatnonzero(control & (split == name)))
        chosen = cells[: n_groups * sizes[name]]
        for group, members in zip(groups, np.array_split(chosen, n_groups)):
            pseudo[members] = group
    adata.obs[args.pseudo_column] = pd.Categorical(pseudo.astype(str))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(output / "hybrid.h5ad")
    composition = pd.crosstab(adata.obs[args.pseudo_column].astype(str).to_numpy(), split)
    composition = composition.loc[groups + [args.control_label]]
    manifest = {
        "input": str(args.input),
        "seed": args.seed,
        "label_column": args.label_column,
        "pseudo_column": args.pseudo_column,
        "control_label": args.control_label,
        "pseudo_share": args.pseudo_share,
        "group_size_by_split": sizes,
        "groups": groups,
        "cells_by_group_and_split": {group: {str(k): int(v) for k, v in row.items()} for group, row in composition.iterrows()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))


def log_cp10k(counts: np.ndarray) -> np.ndarray:
    library = counts.sum(axis=1, keepdims=True)
    return np.log1p(counts / np.maximum(library, 1e-12) * 1e4)


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    ranked = pvalues[order] * len(pvalues) / np.arange(1, len(pvalues) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def wilcoxon_pvalues(group: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pvalues = np.ones(group.shape[1])
    combined = np.vstack([group, reference])
    varying = combined.max(axis=0) > combined.min(axis=0)
    if varying.any():
        result = stats.mannwhitneyu(group[:, varying], reference[:, varying], alternative="two-sided", axis=0, method="asymptotic")
        pvalues[varying] = np.nan_to_num(result.pvalue, nan=1.0)
    return pvalues


def interval(values: np.ndarray, draws: np.ndarray) -> list[float]:
    boot = values[draws].mean(axis=1)
    return [float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]


def evaluate(args: argparse.Namespace) -> None:
    recorded_adata = ad.read_h5ad(args.recorded)
    recorded = dense(recorded_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[recorded_adata.obs_names.astype(str), "split"].to_numpy()
    test = split == "test"
    rows = []
    for seed_dir in args.pseudo_dirs:
        seed_dir = Path(seed_dir)
        manifest = json.loads((seed_dir / "manifest.json").read_text())
        pseudo = ad.read_h5ad(seed_dir / "hybrid.h5ad", backed="r").obs[manifest["pseudo_column"]].astype(str).to_numpy()
        versions = {"recorded": (None, recorded)}
        for fraction in FILL_FRACTIONS:
            for name, root in [("safe_fusion", Path(args.label_free_selector)),
                               ("safe_fusion_true_labels", Path(args.true_label_selector)),
                               ("safe_fusion_pseudo_labels", seed_dir / "selector_condition")]:
                versions[f"{name}@{fraction}"] = (fraction, np.load(root / f"safe_fusion_calibrated_mlp_topk_{fraction}" / "mean.npy", mmap_mode="r"))
        reference = test & (pseudo == manifest["control_label"])
        for version, (fraction, matrix) in versions.items():
            normalized = log_cp10k(np.asarray(matrix[test], dtype=np.float64))
            test_pseudo = pseudo[test]
            reference_rows = normalized[reference[test]]
            for group in manifest["groups"]:
                pvalues = wilcoxon_pvalues(normalized[test_pseudo == group], reference_rows)
                calls = benjamini_hochberg(pvalues) < args.fdr
                rows.append({
                    "seed": manifest["seed"],
                    "group": group,
                    "version": version.split("@")[0],
                    "fill_fraction": fraction or "none",
                    "n_group_cells": int((test_pseudo == group).sum()),
                    "n_reference_cells": int(reference_rows.shape[0]),
                    "n_genes": int(len(pvalues)),
                    "n_discoveries": int(calls.sum()),
                    "false_positive_rate": float(calls.mean()),
                    "min_nominal_p": float(pvalues.min()),
                    "n_nominal_p_below_fdr": int((pvalues < args.fdr).sum()),
                })
    frame = pd.DataFrame(rows)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "pseudo_group_de.csv", index=False)

    frame["unit"] = frame["seed"].astype(str) + ":" + frame["group"]
    units = sorted(frame["unit"].unique())
    rng = np.random.default_rng(args.bootstrap_seed)
    draws = rng.integers(0, len(units), size=(args.draws, len(units)))
    baseline = frame[frame.version == "recorded"].set_index("unit").loc[units]
    summary = {"fdr": args.fdr, "n_pseudo_groups": len(units), "n_genes": int(frame["n_genes"].iloc[0]),
               "group_cells_median": float(frame["n_group_cells"].median()), "reference_cells_median": float(frame["n_reference_cells"].median()),
               "versions": {}}
    for (version, fraction), sub in frame.groupby(["version", "fill_fraction"], sort=False):
        sub = sub.set_index("unit").loc[units]
        fpr = sub["false_positive_rate"].to_numpy()
        nominal = (sub["n_nominal_p_below_fdr"] / sub["n_genes"]).to_numpy()
        entry = {
            "false_positive_rate": interval(fpr, draws),
            "discoveries_per_group": interval(sub["n_discoveries"].to_numpy(dtype=float), draws),
            "groups_with_any_discovery": int((sub["n_discoveries"] > 0).sum()),
            "nominal_p_below_fdr_share": interval(nominal, draws),
        }
        if version != "recorded":
            entry["false_positive_rate_minus_recorded"] = interval(fpr - baseline["false_positive_rate"].to_numpy(), draws)
            entry["nominal_share_minus_recorded"] = interval(nominal - (baseline["n_nominal_p_below_fdr"] / baseline["n_genes"]).to_numpy(), draws)
        summary["versions"][f"{version}@{fraction}"] = entry
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    table = frame.groupby(["version", "fill_fraction"], sort=False).agg(
        false_positive_rate=("false_positive_rate", "mean"),
        discoveries=("n_discoveries", "mean"),
        groups_with_discovery=("n_discoveries", lambda x: int((x > 0).sum())),
        nominal_share=("n_nominal_p_below_fdr", lambda x: float(x.mean())),
    )
    print(table.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--input", default="artifacts/paper_evidence/downstream_deployment/adamson_crispri/hybrid.h5ad")
    prep.add_argument("--splits", default="artifacts/paper_evidence/downstream_deployment/adamson_crispri/splits.parquet")
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--label-column", default="target")
    prep.add_argument("--control-label", default="none")
    prep.add_argument("--pseudo-column", default="pseudo_target")
    prep.add_argument("--pseudo-share", type=float, default=0.5, help="Share of control cells that become pseudo-perturbed.")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--recorded", default="artifacts/paper_evidence/downstream_deployment/adamson_crispri/recorded.h5ad")
    evaluation.add_argument("--splits", default="artifacts/paper_evidence/downstream_deployment/adamson_crispri/splits.parquet")
    evaluation.add_argument("--label-free-selector", default="artifacts/paper_evidence/downstream_deployment/adamson_crispri/selector")
    evaluation.add_argument("--true-label-selector", default="artifacts/paper_evidence/downstream_deployment/adamson_crispri/selector_condition")
    evaluation.add_argument("--pseudo-dirs", nargs="+", required=True)
    evaluation.add_argument("--output-dir", required=True)
    evaluation.add_argument("--fdr", type=float, default=0.05)
    evaluation.add_argument("--draws", type=int, default=2000)
    evaluation.add_argument("--bootstrap-seed", type=int, default=1729)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
