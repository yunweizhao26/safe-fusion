#!/usr/bin/env python3
"""CPA perturbation-response prediction over imputed inputs.

Same experiment as the GEARS runner: for each input matrix (reference truth,
corrupted raw, graph smoothing, dense Safe Fusion, calibrated selective), a
CPA model is trained on development perturbations and predicts held-out
perturbation responses, evaluated against the uncorrupted truth.
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


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

import cpa  # noqa: E402


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
    parser.add_argument("--cells-per-condition", type=int, default=30)
    parser.add_argument("--control-cells", type=int, default=600)
    parser.add_argument("--gene-cap", type=int, default=1500)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=8206)
    parser.add_argument("--test-conditions", nargs="+", required=True)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth_counts = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted_counts = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    conditions_obs = truth_adata.obs["condition"].astype(str).to_numpy()
    controls_obs = truth_adata.obs["control"].astype(int).to_numpy() == 1
    gene_names = (
        truth_adata.var["feature_name"].astype(str).to_numpy()
        if "feature_name" in truth_adata.var
        else np.asarray(truth_adata.var_names, dtype=str)
    )

    # Same cell/gene subset as the GEARS runner for comparability.
    cell_subset: list[int] = []
    for condition in sorted(set(conditions_obs) - {"ctrl"}):
        positions = np.flatnonzero(conditions_obs == condition)
        cell_subset.extend(positions[: args.cells_per_condition].tolist())
    control_positions = np.flatnonzero(controls_obs)
    cell_subset.extend(control_positions[: args.control_cells].tolist())
    cell_subset = np.asarray(sorted(cell_subset), dtype=int)
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
    conditions_obs = conditions_obs[cell_subset]
    controls_obs = controls_obs[cell_subset]
    print("cells:", len(cell_subset), "genes:", len(gene_names))

    test_conditions = sorted(set(args.test_conditions))
    rng = np.random.default_rng(args.seed)
    reference_norm = log1p_cpm(truth_counts)
    reference_ctrl_mean = reference_norm[controls_obs].mean(axis=0)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

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

        obs = truth_adata.obs[["condition", "control"]].iloc[cell_subset].copy()
        obs["condition"] = np.where(
            obs["control"].astype(int) == 1,
            "ctrl",
            obs["condition"].astype(str).str.replace("+ctrl", "", regex=False),
        )
        obs["cell_type"] = "A549"
        obs["split"] = np.where(obs["condition"].isin(test_conditions), "ood", "train")
        train_mask = obs["split"] == "train"
        train_positions = np.flatnonzero(train_mask.to_numpy())
        shuffled = rng.permutation(train_positions)
        n_valid = max(1, int(round(len(shuffled) * 0.15)))
        valid_positions = shuffled[:n_valid]
        obs.loc[obs.index[valid_positions], "split"] = "valid"

        adata = ad.AnnData(
            X=sparse.csr_matrix(matrix),
            obs=obs,
            var=pd.DataFrame({"gene_name": gene_names}),
        )
        adata.obs_names = np.asarray([f"{name}_{i}" for i in range(adata.n_obs)])
        adata.var_names = gene_names

        save_path = output_root / "lightning_logs" / name
        if save_path.exists():
            shutil.rmtree(save_path)

        cpa.CPA.setup_anndata(
            adata,
            perturbation_key="condition",
            control_group="ctrl",
            categorical_covariate_keys=["cell_type"],
            is_count_data=True,
            max_comb_len=1,
        )
        model = cpa.CPA(
            adata=adata,
            split_key="split",
            train_split="train",
            valid_split="valid",
            test_split="ood",
            n_latent=32,
            recon_loss="nb",
            doser_type="linear",
            n_hidden_encoder=128,
            n_layers_encoder=2,
            n_hidden_decoder=128,
            n_layers_decoder=2,
            variational=False,
            seed=args.seed,
        )
        trainer_params = {
            "n_epochs_kl_warmup": None,
            "n_epochs_adv_warmup": 20,
            "n_epochs_mixup_warmup": 5,
            "n_epochs_pretrain_ae": 5,
            "mixup_alpha": 0.1,
            "lr": 1e-4,
            "wd": 1e-6,
            "adv_steps": 3,
            "reg_adv": 10.0,
            "pen_adv": 20.0,
            "adv_lr": 1e-4,
            "adv_wd": 1e-6,
            "n_layers_adv": 2,
            "n_hidden_adv": 128,
            "use_batch_norm_adv": True,
            "use_layer_norm_adv": False,
            "dropout_rate_adv": 0.3,
            "step_size_lr": 25,
            "do_clip_grad": False,
            "adv_loss": "cce",
            "gradient_clip_value": 5.0,
        }
        model.train(
            max_epochs=args.max_epochs,
            use_gpu=True,
            batch_size=256,
            plan_kwargs=trainer_params,
            early_stopping_patience=10,
            check_val_every_n_epoch=5,
            save_path=str(save_path),
        )

        ctrl_adata = adata[adata.obs["condition"] == "ctrl"].copy()
        sampled = ctrl_adata.X[np.random.default_rng(0).choice(ctrl_adata.n_obs, size=adata.n_obs, replace=True), :]
        adata.X = sparse.csr_matrix(sampled)
        model.predict(adata, batch_size=256)
        predicted_raw = np.asarray(adata.obsm["CPA_pred"])
        predicted_cpm = log1p_cpm(np.clip(predicted_raw, 0, None))

        per_condition = []
        for condition in test_conditions:
            condition_cells = conditions_obs == f"{condition}+ctrl"
            if not condition_cells.any():
                continue
            ref_expr = reference_norm[condition_cells].mean(axis=0)
            ref_response = ref_expr - reference_ctrl_mean
            pred_expr = predicted_cpm[condition_cells].mean(axis=0)
            pred_response = pred_expr - reference_ctrl_mean
            valid = np.isfinite(pred_expr) & np.isfinite(ref_expr)
            pearson_expr = float(pearsonr(pred_expr[valid], ref_expr[valid])[0]) if valid.sum() > 2 else float("nan")
            pearson_response = float(pearsonr(pred_response[valid], ref_response[valid])[0]) if valid.sum() > 2 else float("nan")
            overlap = top_k_overlap(pred_response, ref_response)
            rmse = float(np.sqrt(np.mean((pred_response[valid] - ref_response[valid]) ** 2)))
            per_condition.append({
                "method": name, "condition": condition,
                "pearson_expr": pearson_expr, "pearson_response": pearson_response,
                "top100_overlap": overlap, "rmse": rmse,
            })
        frame = pd.DataFrame(per_condition)
        frame.to_parquet(output_root / f"per_condition_{name}.parquet", index=False)
        row = {
            "method": name,
            "n_test_conditions": int(len(frame)),
            "mean_pearson_expr": float(frame["pearson_expr"].mean()),
            "mean_pearson_response": float(frame["pearson_response"].mean()),
            "mean_top100_overlap": float(frame["top100_overlap"].mean()),
            "mean_rmse": float(frame["rmse"].mean()),
        }
        summary_rows.append(row)
        print(json.dumps(row))
        del matrix, adata, model

    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(output_root / "summary.parquet", index=False)
    report = {
        "design": "CPA held-out perturbation-response prediction; reference = uncorrupted truth on all cells of held-out conditions",
        "max_epochs": args.max_epochs,
        "seed": args.seed,
        "test_conditions": test_conditions,
        "results": summary.to_dict(orient="records"),
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
