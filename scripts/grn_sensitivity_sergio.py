#!/usr/bin/env python3
"""GRN inference sensitivity to imputation on SERGIO simulations.

Simulates a small SERGIO network with known regulatory edges, corrupts 10% of
nonzero counts before fitting, runs raw/graph smoothing/dense Safe
Fusion/ALRA, and measures edge AUPRC and top-k overlap of correlation-based
GRN inference against the known edges.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.corruption import corrupt_counts  # noqa: E402
from safefusion_benchmark.downstream import infer_grn_scores  # noqa: E402
from safefusion_benchmark.metrics import average_precision_tie_aware  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-genes", type=int, default=100)
    parser.add_argument("--n-tfs", type=int, default=10)
    parser.add_argument("--n-cells", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--sergio-root", type=Path, help="Checkout containing the SERGIO package")
    args = parser.parse_args()

    if args.sergio_root:
        sys.path.insert(0, str(args.sergio_root))

    from SERGIO.sergio import sergio

    rng = np.random.default_rng(args.seed)
    n_genes, n_tfs, n_cells = args.n_genes, args.n_tfs, args.n_cells
    n_targets = n_genes - n_tfs
    tfs = list(range(n_tfs))
    targets = list(range(n_tfs, n_genes))

    edges: list[tuple[int, int]] = []
    target_rows: list[list[str]] = []
    for target in targets:
        n_regs = int(rng.integers(1, 4))
        regs = rng.choice(tfs, size=n_regs, replace=False).tolist()
        ks = [round(float(x), 3) for x in rng.uniform(0.5, 2.0, size=n_regs)]
        target_rows.append([str(target), str(n_regs), *[str(x) for x in regs], *[str(x) for x in ks]])
        for reg in regs:
            edges.append((target, int(reg)))
    # Ensure every TF regulates at least one target.
    used = set(reg for _, reg in edges)
    for tf in tfs:
        if tf not in used:
            target = int(rng.choice(targets))
            target_rows[target - n_tfs][1] = str(int(target_rows[target - n_tfs][1]) + 1)
            target_rows[target - n_tfs].insert(2, str(tf))
            target_rows[target - n_tfs].insert(2 + len(tfs) + 1, "1.0")
            edges.append((target, tf))

    work = Path(args.output_dir) / "sergio_work"
    work.mkdir(parents=True, exist_ok=True)
    targets_file = work / "targets.csv"
    regs_file = work / "regs.csv"
    with targets_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(target_rows)
    reg_rows = [[str(tf), str(round(float(rng.uniform(0.5, 2.0)), 3))] for tf in tfs]
    with regs_file.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(reg_rows)

    simulator = sergio(
        number_genes=n_genes,
        number_bins=1,
        number_sc=n_cells,
        noise_params=0.1,
        noise_type="dpd",
        decays=0.8,
        sampling_state=10,
    )
    simulator.build_graph(str(targets_file), str(regs_file), shared_coop_state=1)
    simulator.simulate()
    expression = simulator.getExpressions()
    counts = expression[0].T.astype(np.float32)  # cells x genes

    cell_ids = np.asarray([f"cell_{i}" for i in range(n_cells)])
    gene_ids = np.asarray([f"G{i}" for i in range(n_genes)])
    shuffled = rng.permutation(n_cells)
    split = np.empty(n_cells, dtype=object)
    n_dev = int(round(n_cells * 0.70))
    n_val = int(round(n_cells * 0.15))
    split[shuffled[:n_dev]] = "development"
    split[shuffled[n_dev : n_dev + n_val]] = "validation"
    split[shuffled[n_dev + n_val :]] = "test"
    split_frame = pd.DataFrame({"cell_id": cell_ids, "biological_unit": cell_ids, "split": split})
    splits_path = work / "splits.parquet"
    split_frame.to_parquet(splits_path, index=False)

    corrupted, coordinates = corrupt_counts(
        counts, cell_ids, gene_ids, cell_ids, split,
        {"kind": "stratified_nonzero_mask", "fraction": 0.10, "gene_bins": 4, "library_bins": 4},
        args.seed,
    )
    corrupted_path = work / "corrupted.h5ad"
    truth_path = work / "truth.h5ad"
    import anndata as ad
    from scipy import sparse

    obs = pd.DataFrame({"condition": ["ctrl"] * n_cells, "control": np.zeros(n_cells, dtype=int)})
    var = pd.DataFrame({"feature_name": gene_ids})
    adata = ad.AnnData(X=sparse.csr_matrix(counts), obs=obs, var=var)
    adata.obs_names = cell_ids
    adata.var_names = gene_ids
    adata.layers["counts"] = sparse.csr_matrix(counts)
    adata.write_h5ad(truth_path)
    corrupted_adata = ad.AnnData(
        X=sparse.csr_matrix(corrupted),
        obs=obs,
        var=var,
    )
    corrupted_adata.obs_names = cell_ids
    corrupted_adata.var_names = gene_ids
    corrupted_adata.layers["corrupted_counts"] = sparse.csr_matrix(corrupted)
    corrupted_adata.write_h5ad(corrupted_path)
    coordinates.to_parquet(work / "coordinates.parquet", index=False)

    python = sys.executable
    methods: dict[str, Path] = {}
    subprocess.run(
        [python, "scripts/run_leakage_safe_method.py", "--method", "graph_smooth", "--input", str(corrupted_path),
         "--coordinates", str(work / "coordinates.parquet"), "--splits", str(splits_path),
         "--output", str(work / "graph_smooth"), "--seed", str(args.seed)],
        check=True,
    )
    subprocess.run(
        [python, "scripts/run_leakage_safe_method.py", "--method", "safe_fusion", "--input", str(corrupted_path),
         "--coordinates", str(work / "coordinates.parquet"), "--splits", str(splits_path),
         "--output", str(work / "safe_fusion"), "--seed", str(args.seed), "--epochs", "12", "--batch-size", "16"],
        check=True,
    )
    subprocess.run(
        [python, "scripts/run_alra_baseline.py", "--corrupted", str(corrupted_path), "--splits", str(splits_path),
         "--output", str(work / "alra"), "--components", "30", "--power-iterations", "2", "--quantile-prob", "0.001", "--seed", str(args.seed)],
        check=True,
    )
    methods = {
        "corrupted_raw": corrupted,
        "graph_smooth": np.load(work / "graph_smooth/mean.npy", allow_pickle=False),
        "safe_fusion": np.load(work / "safe_fusion/mean.npy", allow_pickle=False),
        "alra": np.load(work / "alra/mean.npy", allow_pickle=False),
    }

    edge_labels = np.zeros((n_genes, n_genes), dtype=bool)
    for target, reg in edges:
        edge_labels[reg, target] = True
    truth_edges = [[gene_ids[reg], gene_ids[target]] for target, reg in edges]
    results = []
    for name, matrix in methods.items():
        labels, scores = infer_grn_scores(np.clip(matrix, 0, None), gene_ids.tolist(), truth_edges)
        prevalence = float(labels.mean())
        auprc = average_precision_tie_aware(labels, scores)
        k = max(1, int(labels.sum()))
        top = np.argsort(-scores)[:k]
        jaccard = float(labels[top].sum() / (2 * k - labels[top].sum()))
        results.append({
            "method": name,
            "edge_auprc": float(auprc),
            "prevalence": prevalence,
            "prevalence_lift": float(auprc / prevalence) if prevalence else float("nan"),
            "topk_jaccard": jaccard,
            "n_edges": int(labels.sum()),
        })
    summary = pd.DataFrame(results)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output / "grn_sensitivity.parquet", index=False)
    report = {
        "design": "SERGIO simulation with known GRN edges; 10% nonzero mask before fitting; correlation-based GRN inference; AUPRC vs known edges",
        "n_genes": n_genes, "n_tfs": n_tfs, "n_cells": n_cells,
        "n_edges": len(edges),
        "results": summary.to_dict(orient="records"),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
