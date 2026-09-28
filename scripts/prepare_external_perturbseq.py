#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense_column(matrix, column: int, rows: np.ndarray) -> np.ndarray:
    value = matrix[:, column]
    if sparse.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=np.float32).ravel()[rows]


def dataset_annotations(adata: ad.AnnData, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    labels = adata.obs["perturbation"].astype("string")
    if dataset == "adamson_crispri":
        control = (
            labels.str.startswith("63(mod)", na=False)
            | labels.str.startswith("Gal4-4", na=False)
        ).to_numpy()
        targets = labels.str.split("_").str[0].fillna("none").to_numpy(dtype=str)
    elif dataset == "dixit_ko":
        if "target" in adata.obs:
            target_series = adata.obs["target"].astype("string").fillna("none")
            targets = target_series.to_numpy(dtype=str)
        else:
            target_series = labels.fillna("none")
            targets = target_series.to_numpy(dtype=str)
        control = (
            labels.eq("control").fillna(False)
            | target_series.str.startswith("INTERGENIC", na=False)
        ).to_numpy()
        targets = np.asarray([value if "_" not in value else "none" for value in targets])
    elif dataset == "papalexi_eccite":
        control = labels.eq("control").fillna(False).to_numpy()
        targets = np.asarray(
            [re.sub(r"g[0-9]+$", "", value) for value in labels.fillna("none").to_numpy(dtype=str)]
        )
    else:
        raise ValueError(dataset)
    targets[control] = "none"
    return targets, control


def stratified_split(
    targets: np.ndarray,
    control: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    split = np.full(len(targets), "excluded", dtype=object)
    groups = ["none", *sorted(set(targets) - {"none"})]
    for group in groups:
        positions = np.flatnonzero(control if group == "none" else targets == group)
        if not len(positions):
            continue
        positions = rng.permutation(positions)
        n_test = max(1, int(round(test_fraction * len(positions))))
        split[positions[:n_test]] = "test"
        split[positions[n_test:]] = "development"
    return split


def variable_genes(counts: sparse.csr_matrix, maximum: int) -> np.ndarray:
    library = np.asarray(counts.sum(axis=1)).ravel()
    scale = np.divide(10_000.0, library, out=np.zeros_like(library), where=library > 0)
    normalized = counts.multiply(scale[:, None]).tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data)
    mean = np.asarray(normalized.mean(axis=0)).ravel()
    second = np.asarray(normalized.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - np.square(mean), 0.0)
    expressed = np.flatnonzero(np.asarray(counts.sum(axis=0)).ravel() > 0)
    return expressed[np.argsort(variance[expressed])[-maximum:]].astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["adamson_crispri", "dixit_ko", "papalexi_eccite"],
    )
    parser.add_argument("--max-conditions", type=int, default=30)
    parser.add_argument("--max-cells-per-condition", type=int, default=100)
    parser.add_argument("--max-control-cells", type=int, default=1000)
    parser.add_argument("--max-genes", type=int, default=2048)
    parser.add_argument(
        "--include-gene",
        nargs="+",
        default=None,
        help="Genes that must be retained in addition to perturbation targets.",
    )
    parser.add_argument("--min-cells-per-split", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.30)
    parser.add_argument("--target-log2fc-max", type=float, default=-0.25)
    parser.add_argument("--target-p-max", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = Path(args.input)
    adata = ad.read_h5ad(source, backed="r")
    counts = adata.X
    genes = adata.var_names.astype(str).to_numpy()
    lookup = {gene: index for index, gene in enumerate(genes)}
    targets, control = dataset_annotations(adata, args.dataset)
    print(f"[prepare] opened {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    rng = np.random.default_rng(args.seed)
    split = stratified_split(targets, control, args.test_fraction, rng)
    development_control = np.flatnonzero(control & (split == "development"))
    if len(development_control) < args.min_cells_per_split:
        raise RuntimeError("too few development control cells")

    qc_rows: list[dict] = []
    for target in sorted(set(targets) - {"none"}):
        development = np.flatnonzero((targets == target) & (split == "development"))
        test = np.flatnonzero((targets == target) & (split == "test"))
        if target not in lookup or min(len(development), len(test)) < args.min_cells_per_split:
            continue
        gene = lookup[target]
        perturbed = dense_column(counts, gene, development)
        reference = dense_column(counts, gene, development_control)
        log2fc = float(np.log2(perturbed.mean() + 1.0) - np.log2(reference.mean() + 1.0))
        _, p_value = mannwhitneyu(perturbed, reference, alternative="two-sided")
        direct_effect_passed = bool(
            log2fc <= args.target_log2fc_max and p_value < args.target_p_max
        )
        passed = direct_effect_passed if args.dataset == "adamson_crispri" else True
        qc_rows.append({
            "target": target,
            "development_cells": int(len(development)),
            "test_cells": int(len(test)),
            "development_target_log2fc": log2fc,
            "development_p_value": float(p_value),
            "direct_target_rna_gate_passed": direct_effect_passed,
            "passed": passed,
        })
    print(f"[prepare] development-only QC evaluated {len(qc_rows)} targets", flush=True)
    qc = pd.DataFrame(qc_rows)
    passed = qc[qc["passed"]].sort_values(
        ["development_target_log2fc", "target"], ascending=[True, True]
    ).head(args.max_conditions)
    if passed.empty:
        raise RuntimeError("no target passed development-only perturbation QC")

    selected: list[int] = []
    per_group = {
        "development": max(1, int(round(args.max_cells_per_condition * (1.0 - args.test_fraction)))),
        "test": max(1, int(round(args.max_cells_per_condition * args.test_fraction))),
    }
    selected_targets = set(passed["target"].astype(str))
    for target in sorted(selected_targets):
        for split_name, cap in per_group.items():
            positions = np.flatnonzero((targets == target) & (split == split_name))
            selected.extend(rng.choice(positions, size=min(cap, len(positions)), replace=False).tolist())
    for split_name, fraction in (("development", 1.0 - args.test_fraction), ("test", args.test_fraction)):
        positions = np.flatnonzero(control & (split == split_name))
        cap = max(1, int(round(args.max_control_cells * fraction)))
        selected.extend(rng.choice(positions, size=min(cap, len(positions)), replace=False).tolist())
    selected = np.asarray(sorted(set(selected)), dtype=np.int64)

    print(f"[prepare] materializing {len(selected)} selected cells", flush=True)
    selected_counts = counts[selected]
    selected_counts = (
        selected_counts.tocsr().astype(np.float32)
        if sparse.issparse(selected_counts)
        else sparse.csr_matrix(selected_counts, dtype=np.float32)
    )
    selected_split = split[selected].astype(str)
    selected_target = targets[selected].astype(str)
    selected_control = control[selected]
    development_rows = np.flatnonzero(selected_split == "development")
    requested_genes = [str(gene) for gene in (args.include_gene or [])]
    missing_requested = sorted(set(requested_genes) - set(lookup))
    if missing_requested:
        raise ValueError(f"requested genes are absent from the source matrix: {missing_requested}")
    target_indices = np.asarray([lookup[target] for target in sorted(selected_targets)], dtype=np.int64)
    required_indices = np.asarray(
        sorted(set(target_indices.tolist()) | {lookup[gene] for gene in requested_genes}),
        dtype=np.int64,
    )
    n_variable = max(1, args.max_genes - len(required_indices))
    feature_indices = variable_genes(selected_counts[development_rows], n_variable)
    print(f"[prepare] selected development-only variable genes", flush=True)
    feature_indices = np.asarray(sorted(set(feature_indices.tolist()) | set(required_indices.tolist())), dtype=np.int64)
    if len(feature_indices) > args.max_genes:
        non_required = [index for index in feature_indices if index not in set(required_indices)]
        feature_indices = np.asarray(
            sorted(required_indices.tolist() + non_required[-(args.max_genes - len(required_indices)):]),
            dtype=np.int64,
        )
    selected_counts = selected_counts[:, feature_indices].tocsr()

    obs = adata.obs.iloc[selected].copy()
    obs["source_cell_id"] = adata.obs_names[selected].astype(str)
    obs["condition"] = np.where(selected_control, "ctrl", selected_target)
    obs["target"] = selected_target
    obs["control"] = selected_control.astype(np.int8)
    obs["preassigned_split"] = selected_split
    obs.index = pd.Index([f"{args.dataset}_{index:05d}" for index in range(len(obs))], name="cell_id")
    var = adata.var.iloc[feature_indices].copy()
    result = ad.AnnData(X=selected_counts, obs=obs, var=var)
    result.layers["counts"] = selected_counts.copy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(destination, compression="gzip")

    report = {
        "source": str(source),
        "source_sha256": sha256(source),
        "dataset": args.dataset,
        "design": "split first; condition QC and HVG selection on development cells only",
        "replicate_limitation": "source has no independent replicate column; perturbation targets are inferential units",
        "cells": int(result.n_obs),
        "genes": int(result.n_vars),
        "conditions": int(len(selected_targets)),
        "condition_targets": sorted(selected_targets),
        "required_genes": requested_genes,
        "split_counts": pd.Series(selected_split).value_counts().to_dict(),
        "control_cells": int(selected_control.sum()),
        "qc_rule": (
            {
                "kind": "development-only direct target RNA knockdown",
                "target_log2fc_max": args.target_log2fc_max,
                "target_p_max": args.target_p_max,
                "min_cells_per_split": args.min_cells_per_split,
            }
            if args.dataset == "adamson_crispri"
            else {
                "kind": "assignment and minimum cells only",
                "reason": "Cas9 KO can disrupt protein without lowering target RNA",
                "direct_target_rna_effect_reported_but_not_used_for_selection": True,
                "min_cells_per_split": args.min_cells_per_split,
            }
        ),
        "development_qc": passed.to_dict(orient="records"),
        "seed": args.seed,
    }
    destination.with_suffix(".report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
