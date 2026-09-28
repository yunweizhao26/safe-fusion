#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
from anndata import read_h5ad
from pandas import DataFrame

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
import SAUCIE
import magic
import scprep
from deepimpute import multinet
import scscope as scScope
import anndata
import scvi
import tensorflow
import torch

REPO = Path(__file__).resolve().parents[2]


def _run_saucie(y, seed):
    tf = importlib.import_module("tensorflow.compat.v1")
    tf.disable_v2_behavior()

    tf.reset_default_graph()
    tf.set_random_seed(seed)
    saucie_model = SAUCIE.SAUCIE(y.shape[1])
    saucie_model.train(SAUCIE.Loader(y, shuffle=True), steps=1000)

    rec_y = saucie_model.get_reconstruction(SAUCIE.Loader(y, shuffle=False))
    return rec_y


def _run_magic(y, seed):
    y_filtered = scprep.filter.filter_rare_genes(y, min_cells=5)
    y_norm = scprep.transform.sqrt(scprep.normalize.library_size_normalize(y_filtered))
    magic_op = magic.MAGIC(t=7, n_pca=20, n_jobs=-1, random_state=seed)
    y_hat = magic_op.fit_transform(y_norm, genes="all_genes")
    return y_hat


def _run_deepimpute(y, seed):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    y_pd = DataFrame(y)
    model = multinet.MultiNet(seed=seed)
    model.fit(y_pd, cell_subset=1, minVMR=0.5)
    imputed_data = model.predict(y_pd)
    return imputed_data.to_numpy()


def _run_scscope(y, seed):
    model = scScope.train(
        y,
        15,
        use_mask=True,
        batch_size=64,
        max_epoch=500,
        epoch_per_check=100,
        T=2,
        exp_batch_idx_input=[],
        encoder_layers=[],
        decoder_layers=[],
        learning_rate=0.0001,
        beta1=0.05,
        num_gpus=1,
        seed=seed,
    )
    _, rec_y, _ = scScope.predict(y, model, batch_effect=[])
    return rec_y


def _run_scvi(y, seed):
    scvi.settings.seed = seed
    adata = anndata.AnnData(y)
    scvi.model.SCVI.setup_anndata(adata)
    model = scvi.model.SCVI(adata)
    model.train()
    return model.get_normalized_expression(return_numpy=True)


def _run_knn_smoothing(y, seed, k=10):
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(y)
    _, indices = nbrs.kneighbors(y)
    smoothed = np.zeros_like(y)
    for i in range(y.shape[0]):
        smoothed[i] = np.mean(y[indices[i]], axis=0)
    return smoothed


METHODS = {
    "SAUCIE": _run_saucie,
    "MAGIC": _run_magic,
    "deepImpute": _run_deepimpute,
    "scScope": _run_scscope,
    "scVI": _run_scvi,
    "knn_smoothing": _run_knn_smoothing,
}


def reset_random_state(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tensorflow.keras.backend.clear_session()
    tensorflow.compat.v1.set_random_seed(seed)
    torch.manual_seed(seed)


def parse_subset(text: str) -> tuple[str, str]:
    disease, sep, tissue = text.partition("|")
    if not sep:
        raise argparse.ArgumentTypeError(f"expected 'disease|tissue', got {text!r}")
    return disease, tissue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--input", required=True, help="CELLxGENE H5AD with disease, tissue and is_primary_data in obs")
    parser.add_argument(
        "--output-root", default=str(REPO / "artifacts/paper_evidence/standard_imputers"),
        help="root of the <method>/<dataset-id>/<disease>/<tissue>.npy outputs",
    )
    parser.add_argument("--dataset-id", help="output directory name of the dataset (default: input file stem)")
    parser.add_argument(
        "--subset", action="append", type=parse_subset,
        help="'disease|tissue' subset to impute; repeat for several (default: every subset)",
    )
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    tensorflow.config.experimental.enable_op_determinism()
    torch.use_deterministic_algorithms(True)

    input_path = Path(args.input)
    dataset_id = args.dataset_id or input_path.stem
    method = METHODS[args.method]

    adata = read_h5ad(input_path)
    adata = adata[adata.obs["is_primary_data"] == True]
    subsets = [tuple(pair) for pair in adata.obs[["disease", "tissue"]].drop_duplicates().to_numpy()]
    if args.subset:
        missing = sorted(set(args.subset) - set(subsets))
        if missing:
            raise ValueError(f"subsets not in {input_path}: {missing}")
        subsets = [pair for pair in subsets if pair in set(args.subset)]

    for number, (disease, tissue) in enumerate(subsets, start=1):
        path = Path(args.output_root) / args.method / dataset_id / disease / f"{tissue}.npy"
        if path.exists():
            print(f"[{number}/{len(subsets)}] {args.method} {disease} | {tissue}: exists, kept", flush=True)
            continue
        mask = (adata.obs["disease"] == disease) & (adata.obs["tissue"] == tissue)
        matrix = adata[mask].X.toarray()
        print(f"[{number}/{len(subsets)}] {args.method} {disease} | {tissue}: {matrix.shape}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        reset_random_state(args.seed)
        np.save(path, method(matrix, args.seed))
        gc.collect()


if __name__ == "__main__":
    main()
