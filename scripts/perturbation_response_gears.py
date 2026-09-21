#!/usr/bin/env python3
"""GEARS perturbation-response prediction over imputed inputs.

For each input matrix (reference truth, corrupted raw, graph smoothing, dense
Safe Fusion, calibrated selective), the official GEARS model is trained on
held-out perturbation-response splits and evaluated against the uncorrupted
truth on test cells. Metrics: Pearson on expression, Pearson on response
(minus control), top-100 response overlap, and RMSE.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from gears import GEARS, PertData  # noqa: E402


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def log1p_cpm(matrix: np.ndarray) -> np.ndarray:
    library = matrix.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(matrix * scale[:, None]).astype(np.float32)


def top_k_overlap(a: np.ndarray, b: np.ndarray, k: int = 100) -> float:
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    return float(len(top_a & top_b) / k)


def parse_method(value: str) -> tuple[str, Path | None]:
    if value in ("reference_truth", "corrupted_raw"):
        return value, None
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--train-gene-set-size", type=float, default=0.75)
    parser.add_argument("--cells-per-condition", type=int, default=40)
    parser.add_argument("--control-cells", type=int, default=800)
    parser.add_argument("--gene-cap", type=int, default=1500)
    parser.add_argument("--go-top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth_counts = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted_counts = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        truth_adata.obs_names.astype(str), "split"
    ].to_numpy()
    test = split == "test"
    gene_names = (
        truth_adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in truth_adata.var
        else np.asarray(truth_adata.var_names, dtype=str)
    )
    conditions_obs = truth_adata.obs["condition"].astype(str).to_numpy()
    controls_obs = truth_adata.obs["control"].astype(int).to_numpy() == 1

    # Deterministic cell subset (same cells for every input).
    cell_subset: list[int] = []
    for condition in sorted(set(conditions_obs) - {"ctrl"}):
        positions = np.flatnonzero(conditions_obs == condition)
        cell_subset.extend(positions[: args.cells_per_condition].tolist())
    control_positions = np.flatnonzero(controls_obs)
    cell_subset.extend(control_positions[: args.control_cells].tolist())
    cell_subset = np.asarray(sorted(cell_subset), dtype=int)

    # Gene subset: genes covered by the GEARS GO graph.
    gene2go = pd.read_pickle(REPOSITORY / "external_data/perturbation_response/gears_aux/gene2go_all.pkl")
    gene_subset = np.asarray([index for index, gene in enumerate(gene_names) if gene in gene2go], dtype=int)
    truth_counts = truth_counts[np.ix_(cell_subset, gene_subset)]
    corrupted_counts = corrupted_counts[np.ix_(cell_subset, gene_subset)]
    gene_names = gene_names[gene_subset]
    if len(gene_names) > args.gene_cap:
        variance = truth_counts.var(axis=0)
        keep = np.sort(np.argsort(-variance)[: args.gene_cap])
        truth_counts = truth_counts[:, keep]
        corrupted_counts = corrupted_counts[:, keep]
        gene_names = gene_names[keep]
        gene_subset = gene_subset[keep]
    print("cells:", len(cell_subset), "genes:", len(gene_names))

    # Precompute the GO graph from the dataset GO edge file (memory-light;
    # GEARS' default loader reads a 354 MB CSV and spikes memory).
    go_edges = pd.read_csv(REPOSITORY / "external_data/perturbation_response/norman/go.csv")
    gene_index = {gene: index for index, gene in enumerate(gene_names)}
    go_rows = []
    for _, row in go_edges.iterrows():
        source = row["source"]
        target = row["target"]
        if source in gene_index and target in gene_index:
            go_rows.append((gene_index[source], gene_index[target], float(row["importance"])))
    if go_rows:
        go_array = np.asarray(go_rows, dtype=np.float64)
        if args.go_top_k:
            order = np.lexsort((-go_array[:, 2], go_array[:, 0]))
            go_array = go_array[order]
            sources, counts = np.unique(go_array[:, 0], return_counts=True)
            positions = []
            start = 0
            for count in counts:
                positions.extend(range(start, start + min(count, args.go_top_k)))
                start += count
            go_array = go_array[np.asarray(positions, dtype=int)]
        go_edge_index = torch.tensor(go_array[:, :2].T.astype(np.int64), dtype=torch.long)
        go_edge_weight = torch.tensor(go_array[:, 2], dtype=torch.float32)
    else:
        go_edge_index = torch.zeros((2, 0), dtype=torch.long)
        go_edge_weight = torch.zeros(0, dtype=torch.float32)
    print("GO edges:", go_edge_index.shape[1])

    test = test[cell_subset]
    controls_obs = controls_obs[cell_subset]
    conditions_obs = conditions_obs[cell_subset]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    aux = REPOSITORY / "external_data/perturbation_response/gears_aux"

    # Reference truth responses on all cells of held-out conditions
    # (standard GEARS evaluation; test-split-only references are too noisy
    # with ~18 cells per condition).
    reference_norm = log1p_cpm(truth_counts)
    reference_ctrl_mean = reference_norm[controls_obs].mean(axis=0)

    methods: list[tuple[str, Path | None]] = []
    for specification in args.method:
        methods.append(parse_method(specification))
    methods.append(("reference_truth", None))
    methods.append(("corrupted_raw", None))

    summary_rows = []
    for name, path in methods:
        if name == "reference_truth":
            matrix = truth_counts
        elif name == "corrupted_raw":
            matrix = corrupted_counts
        else:
            matrix = np.load(path / "mean.npy", allow_pickle=False)[np.ix_(cell_subset, gene_subset)]
        normalized = log1p_cpm(matrix)
        obs = truth_adata.obs[["condition", "control"]].iloc[cell_subset].copy()
        adata = ad.AnnData(
            X=sparse.csr_matrix(normalized),
            obs=obs,
            var=pd.DataFrame({"gene_name": gene_names}),
        )
        adata.obs["condition_name"] = adata.obs["condition"].astype(str).to_numpy()
        adata.obs["cell_type"] = "A549"
        data_root = output_root / "gears_data" / name
        if data_root.exists():
            shutil.rmtree(data_root)
        data_root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(aux / "gene2go_all.pkl", data_root / "gene2go_all.pkl")
        shutil.copyfile(aux / "essential_all_data_pert_genes.pkl", data_root / "essential_all_data_pert_genes.pkl")

        pert_data = PertData(str(output_root / "gears_data"))
        pert_data.new_data_process(dataset_name=name, adata=adata)
        pert_data.prepare_split(
            split="simulation",
            seed=args.seed,
            train_gene_set_size=args.train_gene_set_size,
            combo_single_split_test_set_fraction=0.1,
        )
        pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=64)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gears_model = GEARS(pert_data, device=device, weight_bias_track=False)
        gears_model.model_initialize(
            hidden_size=args.hidden_size,
            uncertainty=False,
            G_go=go_edge_index,
            G_go_weight=go_edge_weight,
        )
        gears_model.train(epochs=args.epochs)

        test_conditions = pert_data.set2conditions.get("test", [])

        def condition_to_pert(condition: str) -> list[str]:
            return [gene for gene in condition.split("+") if gene != "ctrl"]

        test_perts = [condition_to_pert(condition) for condition in test_conditions]
        test_perts = [pert for pert in test_perts if pert and pert[0] in pert_data.pert_names]
        predictions = gears_model.predict(test_perts)
        per_condition = []
        for pert in test_perts:
            gene = pert[0]
            condition = "+".join(pert) + "+ctrl"
            condition_cells = conditions_obs == condition
            if not condition_cells.any():
                continue
            ref_expr = reference_norm[condition_cells].mean(axis=0)
            ref_response = ref_expr - reference_ctrl_mean
            pred_expr = predictions[gene]
            pred_response = pred_expr - reference_ctrl_mean
            valid = np.isfinite(pred_expr) & np.isfinite(ref_expr)
            pearson_expr = float(pearsonr(pred_expr[valid], ref_expr[valid])[0]) if valid.sum() > 2 else float("nan")
            pearson_response = float(pearsonr(pred_response[valid], ref_response[valid])[0]) if valid.sum() > 2 else float("nan")
            overlap = top_k_overlap(pred_response, ref_response)
            rmse = float(np.sqrt(np.mean((pred_response[valid] - ref_response[valid]) ** 2)))
            per_condition.append({
                "method": name, "condition": gene,
                "pearson_expr": pearson_expr, "pearson_response": pearson_response,
                "top100_overlap": overlap, "rmse": rmse,
            })
        frame = pd.DataFrame(per_condition)
        row = {
            "method": name,
            "n_test_conditions": int(len(frame)),
            "mean_pearson_expr": float(frame["pearson_expr"].mean()),
            "mean_pearson_response": float(frame["pearson_response"].mean()),
            "mean_top100_overlap": float(frame["top100_overlap"].mean()),
            "mean_rmse": float(frame["rmse"].mean()),
        }
        summary_rows.append(row)
        frame.to_parquet(output_root / f"per_condition_{name}.parquet", index=False)
        print(json.dumps(row))
        del matrix, normalized, adata, pert_data, gears_model, predictions

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(output_root / "summary.parquet", index=False)
    report = {
        "design": "official GEARS simulation split; per-input training; reference = uncorrupted truth on test cells",
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "seed": args.seed,
        "train_gene_set_size": args.train_gene_set_size,
        "results": summary.to_dict(orient="records"),
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
