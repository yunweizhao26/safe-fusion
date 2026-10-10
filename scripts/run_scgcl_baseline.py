#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo", default="baselines_and_data/scGCL")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-genes", type=int, default=2048)
    parser.add_argument(
        "--target-sum",
        type=float,
        default=None,
        help=(
            "Library size after normalization. The default uses the median "
            "corrupted library, matching upstream scanpy.normalize_per_cell."
        ),
    )
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--num-centroids", type=int, default=9)
    parser.add_argument("--num-kmeans", type=int, default=5)
    parser.add_argument("--cluster-iters", type=int, default=20)
    parser.add_argument("--pred-hidden", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--stability-note",
        default=None,
        help="Optional disclosed deviation used only to make the pinned model finite.",
    )
    parser.add_argument("--ema", type=float, default=0.9)
    parser.add_argument(
        "--seed", type=int, default=0, help="Upstream scGCL fixes NumPy/Torch seed 0."
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume exactly from OUTPUT/checkpoint.pt after preemption.",
    )
    return parser.parse_args()


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalization_target(counts: np.ndarray, genes: np.ndarray) -> float:
    library = counts[:, genes].sum(axis=1)
    positive = library[library > 0]
    if not len(positive):
        raise ValueError("scGCL input has no positive cell libraries")
    if len(positive) != counts.shape[0]:
        raise ValueError("upstream scGCL would filter empty cells; none may be dropped here")
    return float(np.median(positive))


def preprocess(
    counts: np.ndarray,
    genes: np.ndarray,
    target_sum: float | None = None,
) -> np.ndarray:
    expressed = np.flatnonzero(np.sum(counts, axis=0) > 0)
    target = normalization_target(counts, expressed) if target_sum is None else target_sum
    if target <= 0:
        raise ValueError("normalization target must be positive")
    library = counts[:, expressed].sum(axis=1, keepdims=True)
    scale = target / library
    return np.log1p(counts[:, genes] * scale).astype(np.float32)


def choose_genes(counts: np.ndarray, maximum: int) -> np.ndarray:
    expressed = np.flatnonzero(np.sum(counts, axis=0) > 0)
    if len(expressed) <= maximum:
        return expressed.astype(np.int64)
    import scanpy as sc

    normalized = preprocess(counts, expressed)
    work = ad.AnnData(normalized)
    sc.pp.highly_variable_genes(
        work,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        n_top_genes=maximum,
        subset=False,
    )
    selected = np.flatnonzero(work.var["highly_variable"].to_numpy())
    if len(selected) != maximum:
        raise RuntimeError(
            f"upstream HVG selection returned {len(selected)} genes, expected {maximum}"
        )
    return expressed[selected].astype(np.int64)


def rescale_embedding(
    embedding: np.ndarray,
    corrupted_counts: np.ndarray,
    genes: np.ndarray,
    normalized_input: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.maximum(np.asarray(embedding, dtype=np.float32), 0.0)
    if normalized_input is None:
        normalized_input = preprocess(corrupted_counts, genes)
    log_norm = np.linalg.norm(normalized_input, axis=1, keepdims=True)
    reconstructed_log1p = embedding * log_norm
    relative_counts = np.expm1(reconstructed_log1p.astype(np.float64))
    library = corrupted_counts[:, genes].sum(axis=1, keepdims=True).astype(np.float64)
    denominator = relative_counts.sum(axis=1, keepdims=True)
    selected = np.divide(
        relative_counts * library,
        denominator,
        out=np.asarray(corrupted_counts[:, genes], dtype=np.float64).copy(),
        where=denominator > 0,
    )
    output = np.asarray(corrupted_counts, dtype=np.float32).copy()
    output[:, genes] = selected.astype(np.float32)
    return output, reconstructed_log1p.astype(np.float32)


def graph_tensors(
    x: np.ndarray,
    neighbors: int,
    pca_components: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_components = max(2, min(pca_components, x.shape[0] - 1, x.shape[1]))
    representation = PCA(n_components=n_components, random_state=0).fit_transform(x)
    k = max(1, min(neighbors, x.shape[0] - 1))
    adjacency = kneighbors_graph(
        representation, k, mode="connectivity", metric="cosine", include_self=True
    ).tocsr()
    undirected = (adjacency + adjacency.T).astype(bool).astype(np.float32).tocsr()
    undirected.setdiag(0.0)
    undirected.eliminate_zeros()
    encoder_graph = undirected.tocoo()
    edge_index = torch.as_tensor(
        np.vstack([encoder_graph.row, encoder_graph.col]),
        dtype=torch.long,
        device=device,
    )
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32, device=device)
    neighbor_graph = undirected.copy()
    neighbor_graph.setdiag(1.0)
    neighbor_graph = neighbor_graph.tocoo()
    neighbor_index = torch.as_tensor(
        np.vstack([neighbor_graph.row, neighbor_graph.col]),
        dtype=torch.long,
        device=device,
    )
    neighbor_weight = torch.ones(
        neighbor_index.shape[1], dtype=torch.float32, device=device
    )
    return edge_index, edge_weight, neighbor_index, neighbor_weight


def zinb_loss(
    counts: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    dropout: torch.Tensor,
) -> torch.Tensor:
    eps = 1e-10
    theta = torch.clamp(dispersion, 1e-4, 1e6)
    mu = torch.clamp(mean, 1e-5, 1e6)
    pi = torch.clamp(dropout, eps, 1.0 - eps)
    nb = (
        torch.lgamma(theta + eps)
        + torch.lgamma(counts + 1.0)
        - torch.lgamma(counts + theta + eps)
        + (theta + counts) * torch.log1p(mu / (theta + eps))
        + counts * (torch.log(theta + eps) - torch.log(mu + eps))
        - torch.log1p(-pi + eps)
    )
    zero_nb = torch.pow(theta / (theta + mu + eps), theta)
    zero = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
    return torch.where(counts < 1e-8, zero, nb).sum()


def device_safe_neighbor_indices(
    neighbor,
    adjacency: torch.Tensor,
    student: torch.Tensor,
    teacher: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    import faiss

    n_data, dimension = student.shape
    device = student.device
    similarity = student @ teacher.T.detach()
    similarity += torch.eye(n_data, device=device) * 10
    _, knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)

    cell_index = torch.arange(n_data, device=device)
    knn_graph = neighbor.create_sparse(knn)
    locality = knn_graph * adjacency

    teacher_numpy = teacher.detach().cpu().numpy()
    labels = []
    for seed in range(neighbor.num_kmeans):
        kmeans = faiss.Kmeans(
            dimension,
            neighbor.num_centroids,
            niter=neighbor.clus_num_iters,
            gpu=False,
            seed=seed + 1234,
        )
        kmeans.train(teacher_numpy)
        _, assignment = kmeans.index.search(teacher_numpy, 1)
        labels.append(assignment[:, 0])
    cluster_labels = torch.as_tensor(np.stack(labels), device=device).float()
    close = None
    for each_k_idx in range(neighbor.num_kmeans):
        own = cluster_labels[each_k_idx, cell_index].unsqueeze(1).expand(-1, top_k)
        candidate = cluster_labels[each_k_idx, knn]
        current = own == candidate
        close = current if close is None else close | current

    globality = neighbor.create_sparse_revised(knn, close)
    return (locality + globality).coalesce().indices()


def forward_loss(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    neighbor_index: torch.Tensor,
    neighbor_weight: torch.Tensor,
    epoch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    student = model.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
    prediction = model.student_predictor(student)
    decoded = model.ZINB_Encoder(student)
    dropout = model.pi_Encoder(decoded)
    dispersion = torch.clamp(model.disp_Encoder(decoded), 1e-4, 1e4).detach()
    mean = torch.clamp(torch.exp(model.mean_Encoder(decoded)), 1e-5, 1e6).detach()
    with torch.no_grad():
        teacher = model.teacher_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)

    adjacency = torch.sparse_coo_tensor(
        neighbor_index,
        neighbor_weight,
        size=(x.shape[0], x.shape[0]),
        device=x.device,
    ).coalesce()
    indices = device_safe_neighbor_indices(
        model.neighbor,
        adjacency,
        F.normalize(student, dim=-1, p=2),
        F.normalize(teacher, dim=-1, p=2),
        model.topk,
    )
    contrast = (
        2.0
        - 2.0
        * (
            F.normalize(prediction[indices[0]], dim=-1, p=2)
            * F.normalize(teacher[indices[1]].detach(), dim=-1, p=2)
        ).sum(dim=-1)
    ).mean()
    contrast += (
        2.0
        - 2.0
        * (
            F.normalize(prediction[indices[1]], dim=-1, p=2)
            * F.normalize(teacher[indices[0]].detach(), dim=-1, p=2)
        ).sum(dim=-1)
    ).mean()
    reconstruction = F.mse_loss(student, x)
    return student, zinb_loss(x, mean, dispersion, dropout) + contrast + reconstruction


def checkpoint_config(args: argparse.Namespace, source: Path) -> dict[str, object]:
    return {
        "source": str(source.resolve()),
        "epochs": args.epochs,
        "max_genes": args.max_genes,
        "target_sum": args.target_sum,
        "neighbors": args.neighbors,
        "pca_components": args.pca_components,
        "topk": args.topk,
        "num_centroids": args.num_centroids,
        "num_kmeans": args.num_kmeans,
        "cluster_iters": args.cluster_iters,
        "pred_hidden": args.pred_hidden,
        "lr": args.lr,
        "stability_note": args.stability_note,
        "ema": args.ema,
        "seed": args.seed,
    }


def save_training_checkpoint(
    path: Path,
    model,
    optimizer,
    completed_epochs: int,
    losses: list[float],
    config: dict[str, object],
) -> None:
    payload = {
        "format": 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "completed_epochs": completed_epochs,
        "losses": losses,
        "ema_step": model.teacher_ema_updater.step,
        "config": config,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_training_checkpoint(
    path: Path,
    model,
    optimizer,
    expected_config: dict[str, object],
    device: torch.device,
) -> tuple[int, list[float]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != 1:
        raise ValueError(f"unsupported scGCL checkpoint format: {payload.get('format')}")
    if payload.get("config") != expected_config:
        raise ValueError("scGCL checkpoint configuration does not match this run")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    model.teacher_ema_updater.step = int(payload["ema_step"])
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if device.type == "cuda" and payload.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    completed = int(payload["completed_epochs"])
    losses = [float(value) for value in payload["losses"]]
    if completed != len(losses):
        raise ValueError("scGCL checkpoint epoch/loss history is inconsistent")
    return completed, losses


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for scGCL; run this adapter with .venv-baselines"
        )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    repo = Path(args.repo).resolve()
    if not (repo / "models" / "AFGRL.py").exists():
        raise FileNotFoundError(f"official scGCL checkout is incomplete: {repo}")
    sys.path.insert(0, str(repo))
    from models.AFGRL import AFGRL

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    corrupted_path = Path(args.corrupted)
    adata = ad.read_h5ad(corrupted_path)
    layer = "corrupted_counts" if "corrupted_counts" in adata.layers else "counts"
    counts = dense(adata.layers[layer]).astype(np.float32)
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("scGCL input counts must be finite and nonnegative")
    if not np.allclose(counts, np.ceil(counts)):
        raise ValueError("scGCL input must contain integer UMI counts")
    counts = np.ceil(counts).astype(np.float32)
    genes = choose_genes(counts, args.max_genes)
    x_numpy = preprocess(counts, genes, args.target_sum)
    actual_target_sum = (
        normalization_target(counts, np.flatnonzero(counts.sum(axis=0) > 0))
        if args.target_sum is None
        else float(args.target_sum)
    )
    x = torch.as_tensor(x_numpy, dtype=torch.float32, device=device)
    edge_index, edge_weight, neighbor_index, neighbor_weight = graph_tensors(
        x_numpy, args.neighbors, args.pca_components, device
    )

    model_args = SimpleNamespace(
        dropout=0.0,
        pred_hid=args.pred_hidden,
        mad=args.ema,
        epochs=args.epochs,
        topk=min(args.topk, x.shape[0]),
        num_centroids=min(args.num_centroids, x.shape[0]),
        num_kmeans=args.num_kmeans,
        clus_num_iters=args.cluster_iters,
    )
    model = AFGRL([x.shape[1], x.shape[1]], model_args).to(device)
    model._device = device
    model.neighbor.device = device
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    run_config = checkpoint_config(args, corrupted_path)
    start_epoch = 0
    losses: list[float] = []
    if args.resume:
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume requested but checkpoint is missing: {checkpoint}")
        start_epoch, losses = load_training_checkpoint(
            checkpoint, model, optimizer, run_config, device
        )
        print(f"[scGCL] resumed after epoch {start_epoch}/{args.epochs}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        student, loss = forward_loss(
            model, x, edge_index, edge_weight, neighbor_index, neighbor_weight, epoch
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_moving_average()
        value = float(loss.detach().cpu())
        losses.append(value)
        print(f"[scGCL] epoch={epoch + 1}/{args.epochs} loss={value:.6f}", flush=True)
        if args.checkpoint_every > 0 and (epoch + 1) % args.checkpoint_every == 0:
            save_training_checkpoint(
                checkpoint, model, optimizer, epoch + 1, losses, run_config
            )

    model.eval()
    with torch.no_grad():
        embedding = model.student_encoder(x=x, edge_index=edge_index, edge_weight=edge_weight)
        embedding = F.normalize(embedding, dim=-1, p=2).cpu().numpy().astype(np.float32)
    mean, reconstructed_log1p = rescale_embedding(embedding, counts, genes, x_numpy)
    np.save(output / "embedding.npy", embedding, allow_pickle=False)
    np.save(output / "reconstructed_log1p.npy", reconstructed_log1p, allow_pickle=False)
    np.save(output / "selected_gene_indices.npy", genes, allow_pickle=False)
    np.save(output / "mean.npy", mean, allow_pickle=False)
    np.save(output / "variance.npy", np.ones_like(mean, dtype=np.float32), allow_pickle=False)
    metadata = {
        "method": (
            "scgcl_official_compatibility_stability"
            if args.stability_note
            else "scgcl_official_standard_usage"
        ),
        "official_repo": str(repo),
        "official_commit": "317015acdf06d2929c20a7d2858bac539b3d8ebd",
        "source": str(corrupted_path),
        "source_sha256": sha256(corrupted_path),
        "input_layer": layer,
        "shape": [int(v) for v in mean.shape],
        "selected_genes": int(len(genes)),
        "normalization": "upstream normalize_per_cell median library, log1p, Scanpy HVG",
        "normalization_target": actual_target_sum,
        "graph": "upstream 15-NN cosine PCA graph, undirected unit edge weights",
        "runtime_versions": {
            "anndata": version("anndata"),
            "faiss_cpu": version("faiss-cpu"),
            "numpy": version("numpy"),
            "scanpy": version("scanpy"),
            "torch": version("torch"),
            "torch_geometric": version("torch-geometric"),
        },
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "upstream_default_learning_rate": 1e-3,
        "stability_note": args.stability_note,
        "seed": args.seed,
        "device": str(device),
        "transductive": True,
        "heldout_projection_api": False,
        "labels_used_for_fit": False,
        "masked_truth_used_for_fit": False,
        "postprocessing": (
            "official unit-norm embedding multiplied by corrupted log1p-input L2 norm, "
            "expm1 inverted, and selected-gene block rescaled to its corrupted library mass"
        ),
        "final_loss": losses[-1],
        "losses": losses,
        "resumed_from_epoch": start_epoch,
        "cell_ids": adata.obs_names.astype(str).tolist(),
        "gene_ids": adata.var_names.astype(str).tolist(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
