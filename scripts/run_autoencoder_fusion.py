#!/usr/bin/env python3
"""Autoencoder fusion network (the original Safe Fusion value model), trained leak-free.

The network combines an autoencoder expression prior with precision-weighted
teacher proposals. No training target is visible to any input. Two modes:

- resampled: each epoch hides a fresh 15% of the recorded nonzero entries of the
  model-fitting cells. The gene median, SVD and kNN teacher proposals and the
  cell representation are recomputed from the hidden matrix (three teachers).
- masked_positives: the training targets are the masked positives of the
  model-fitting cells, which are zero in every input. Teacher proposals come
  from contracts whose fitting-cell proposals exclude the cell's own counts, so
  any teacher set can be used. These are the targets of the stacked value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))
sys.path.insert(0, str(REPOSITORY))  # the repository's fusion/ package

from run_leakage_safe_method import GraphTeacher, SVDTeacher, fill_zeros, gene_medians, log1p_cpm  # noqa: E402
from safefusion_benchmark.contracts import order_hash, write_output_contract  # noqa: E402
from safefusion_benchmark.hashing import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["resampled", "masked_positives"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-contract", action="append", default=[])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch

    from fusion.eval import predict_latent_truth
    from fusion.train import TrainConfig, train_latent_truth

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    adata = ad.read_h5ad(args.input)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    training = np.isin(split, ["development", "validation"])
    fit_rows = np.flatnonzero(training)
    fit_counts = counts[training]
    normalized, _ = log1p_cpm(counts)
    n_pca = min(30, len(fit_rows) - 1, counts.shape[1] - 1)
    feature_pca = PCA(n_components=n_pca, svd_solver="randomized", random_state=args.seed).fit(normalized[training])
    fit_pca = feature_pca.transform(normalized[training]).astype(np.float32)
    all_pca = feature_pca.transform(normalized).astype(np.float32)

    if args.mode == "resampled":
        medians = gene_medians(counts, training)
        svd = SVDTeacher(counts, training, args.seed, 50)
        graph = GraphTeacher(counts, training, svd.model, 30)
        teachers = {"gene_median": fill_zeros(counts, medians), "svd_impute": svd.proposals(counts), "graph_smooth": graph.proposals(counts)}
        targets = fit_counts

        def epoch_inputs(epoch: int):
            rng = np.random.default_rng([args.seed, epoch])
            mask = (fit_counts > 0) & (rng.random(fit_counts.shape, dtype=np.float32) < 0.15)
            hidden = np.where(mask, 0.0, fit_counts).astype(np.float32)
            stack = np.stack([fill_zeros(hidden, medians), svd.training_proposals(hidden), graph.training_proposals(hidden)])
            return hidden, mask.astype(np.float32), stack, feature_pca.transform(log1p_cpm(hidden)[0]).astype(np.float32)
    else:
        if len(args.teacher_contract) < 2:
            raise ValueError("masked_positives mode needs at least two --teacher-contract directories")
        teachers = {}
        for path in args.teacher_contract:
            meta = json.loads((Path(path) / "metadata.json").read_text())
            if meta.get("scale") != "counts" or not meta["parameters"].get("fitting_cell_proposals_exclude_own_counts"):
                raise ValueError(f"teacher {path} must be a count-scale contract whose fitting-cell proposals exclude own counts")
            teachers[Path(path).name] = np.maximum(np.load(Path(path) / "mean.npy"), 0.0).astype(np.float32)
        coordinates = pd.read_parquet(args.coordinates)
        rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
        cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
        keep = training[rows]
        position = np.full(len(counts), -1, dtype=np.int64)
        position[fit_rows] = np.arange(len(fit_rows))
        targets = fit_counts.copy()
        targets[position[rows[keep]], cols[keep]] = coordinates["original_value"].to_numpy(dtype=np.float32)[keep]
        mask = np.zeros_like(fit_counts)
        mask[position[rows[keep]], cols[keep]] = 1.0
        stack = np.stack([teachers[name][training] for name in teachers])

        def epoch_inputs(epoch: int):
            return fit_counts, mask, stack, fit_pca

    gene_mean = np.log1p(np.mean(fit_counts, axis=0)).astype(np.float32)
    gene_dropout = np.mean(fit_counts <= 0, axis=0).astype(np.float32)
    config = TrainConfig(
        batch_size=args.batch_size, epochs=args.epochs, lr=1e-3, mask_fraction=0.15,
        teacher_weight=1.0, best_teacher_weight=0.8, best_teacher_min_log=0.0, best_teacher_temp=0.5,
        teacher_warmup_epochs=1, teacher_ramp_epochs=3, teacher_dropout=0.4, teacher_loss_on_prior=True,
        teacher_calibration_weight=1e-3, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model, history = train_latent_truth(
        targets, {name: value[training] for name, value in teachers.items()}, config,
        gene_mean=gene_mean, gene_dropout=gene_dropout, pca_features=fit_pca, pca_proj_dim=8,
        seed=args.seed, epoch_inputs=epoch_inputs,
    )
    prediction, _, variance_log = predict_latent_truth(
        model, counts, teachers=teachers, fuse=True, cell_loglib=np.log1p(counts.sum(axis=1)).astype(np.float32),
        gene_mean=gene_mean, gene_dropout=gene_dropout, pca_features=all_pca, batch_size=args.batch_size, device=config.device,
    )
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    metadata = {
        "method": f"autoencoder_fusion_{args.mode}",
        "scale": "counts",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["development", "validation"],
        "parameters": {
            "mode": args.mode,
            "teacher_names": list(teachers),
            "train_config": config.__dict__,
            "history": history,
            "fit_cells": int(training.sum()),
            "test_used_for_fit": False,
            "training_targets_hidden_from_all_inputs": True,
            "fitting_cell_proposals_exclude_own_counts": True,
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.input),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, prediction.astype(np.float32), metadata,
                          variance=(variance_log * np.square(1.0 + prediction)).astype(np.float32))
    print(json.dumps({"mode": args.mode, "teachers": list(teachers), "val_loss": history["val_loss"]}))


if __name__ == "__main__":
    main()
