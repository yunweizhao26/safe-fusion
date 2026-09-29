#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file
from safefusion_benchmark.splits import FOLDS, training_folds


def log1p_cpm(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    library = counts.sum(axis=1, dtype=np.float64)
    scale = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(counts * scale[:, None]).astype(np.float32), library.astype(np.float32)


def gene_medians(counts: np.ndarray, training: np.ndarray) -> np.ndarray:
    result = np.zeros(counts.shape[1], dtype=np.float32)
    fit = counts[training]
    for gene in range(counts.shape[1]):
        values = fit[fit[:, gene] > 0, gene]
        if len(values):
            result[gene] = np.median(values)
    return result


def fill_zeros(counts: np.ndarray, medians: np.ndarray) -> np.ndarray:
    output = counts.astype(np.float32, copy=True)
    rows, cols = np.where(output == 0)
    output[rows, cols] = medians[cols]
    return output


def gene_median(counts: np.ndarray, training: np.ndarray) -> np.ndarray:
    return fill_zeros(counts, gene_medians(counts, training))


def fit_pca(normalized: np.ndarray, components: int, seed: int) -> PCA:
    n_components = min(components, normalized.shape[0] - 1, normalized.shape[1] - 1)
    return PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(normalized)


def svd_reconstruct(model: PCA, counts: np.ndarray) -> np.ndarray:
    normalized, library = log1p_cpm(counts)
    reconstructed = model.inverse_transform(model.transform(normalized))
    reconstructed = np.expm1(np.clip(reconstructed, 0.0, 20.0))
    return np.clip(reconstructed * (library[:, None] / 1e4), 0.0, None).astype(np.float32)


class SVDTeacher:
    def __init__(self, counts: np.ndarray, training: np.ndarray, seed: int, components: int):
        normalized, _ = log1p_cpm(counts)
        self.training_rows = np.flatnonzero(training)
        self.model = fit_pca(normalized[training], components, seed)
        self.folds = training_folds(len(self.training_rows), seed)
        self.fold_models = [
            fit_pca(normalized[self.training_rows[self.folds != fold]], components, seed)
            for fold in range(FOLDS)
        ]

    def training_proposals(self, fit_counts: np.ndarray) -> np.ndarray:
        output = np.empty_like(fit_counts, dtype=np.float32)
        for fold, model in enumerate(self.fold_models):
            rows = self.folds == fold
            output[rows] = svd_reconstruct(model, fit_counts[rows])
        return output

    def proposals(self, counts: np.ndarray) -> np.ndarray:
        output = svd_reconstruct(self.model, counts)
        output[self.training_rows] = self.training_proposals(counts[self.training_rows])
        return output


class GraphTeacher:
    def __init__(self, counts: np.ndarray, training: np.ndarray, pca: PCA, neighbors: int):
        self.pca = pca
        self.training_rows = np.flatnonzero(training)
        self.fit_counts = counts[training]
        fit_embedding = self.embed(self.fit_counts)
        self.k = min(neighbors, len(fit_embedding) - 1)
        self.index = NearestNeighbors(n_neighbors=self.k + 1, metric="euclidean").fit(fit_embedding)
        distances, _ = self.neighbours(fit_embedding, np.arange(len(fit_embedding)))
        positive = distances[distances > 0]
        self.bandwidth = max(float(np.median(positive)) if len(positive) else 1.0, 1e-6)

    def embed(self, counts: np.ndarray) -> np.ndarray:
        normalized, _ = log1p_cpm(counts)
        return self.pca.transform(normalized)

    def neighbours(self, embedding: np.ndarray, self_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        distances, indices = self.index.kneighbors(embedding, n_neighbors=self.k + 1)
        drop = indices == self_index[:, None]
        drop[~drop.any(axis=1), -1] = True
        keep = ~drop
        return distances[keep].reshape(len(indices), self.k), indices[keep].reshape(len(indices), self.k)

    def smooth(self, counts: np.ndarray, self_index: np.ndarray) -> np.ndarray:
        distances, indices = self.neighbours(self.embed(counts), self_index)
        weights = np.exp(-np.square(distances / self.bandwidth))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        output = np.empty(counts.shape, dtype=np.float32)
        for start in range(0, len(counts), 128):
            stop = min(start + 128, len(counts))
            output[start:stop] = np.einsum(
                "nk,nkg->ng", weights[start:stop], self.fit_counts[indices[start:stop]], optimize=True
            )
        return output

    def training_proposals(self, fit_counts: np.ndarray) -> np.ndarray:
        return self.smooth(fit_counts, np.arange(len(fit_counts)))

    def proposals(self, counts: np.ndarray) -> np.ndarray:
        self_index = np.full(len(counts), -1)
        self_index[self.training_rows] = np.arange(len(self.training_rows))
        return self.smooth(counts, self_index)


class ConditionGraphTeacher:
    def __init__(self, counts: np.ndarray, training: np.ndarray, conditions: np.ndarray, pca: PCA, neighbors: int):
        conditions = np.asarray(conditions).astype(str)
        self.groups = {}
        for condition in np.unique(conditions):
            rows = np.flatnonzero(conditions == condition)
            if training[rows].sum() < 2:
                raise ValueError(f"condition {condition} has fewer than two model-fitting cells")
            self.groups[condition] = (rows, GraphTeacher(counts[rows], training[rows], pca, neighbors))

    def proposals(self, counts: np.ndarray) -> np.ndarray:
        output = np.empty(counts.shape, dtype=np.float32)
        for rows, teacher in self.groups.values():
            output[rows] = teacher.proposals(counts[rows])
        return output


VALUE_MODELS = ("boosted", "linear")
MAX_VALUE_FIT_ENTRIES = 600_000


def value_features(teacher_logs: np.ndarray, gene_mean: np.ndarray, detection: np.ndarray,
                   library_log: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [teacher_logs, teacher_logs.std(axis=1), gene_mean[cols], detection[cols], library_log[rows]]
    ).astype(np.float32)


def fit_value_model(kind: str, teacher_logs: np.ndarray, features: np.ndarray, target: np.ndarray, seed: int):
    if kind == "linear":
        coefficients, *_ = np.linalg.lstsq(
            np.column_stack([teacher_logs, np.ones(len(target))]).astype(np.float64), target, rcond=None
        )
        return kind, coefficients
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(target), size=min(len(target), MAX_VALUE_FIT_ENTRIES), replace=False))
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
        min_samples_leaf=100, early_stopping=True, validation_fraction=0.1, random_state=seed,
    )
    model.fit(features[keep], target[keep])
    return kind, model


def predict_value_model(fitted, teacher_logs: np.ndarray, features: np.ndarray) -> np.ndarray:
    kind, model = fitted
    if kind == "linear":
        return np.column_stack([teacher_logs, np.ones(len(teacher_logs))]) @ model
    return model.predict(features)


def fused_value(
    counts: np.ndarray,
    training: np.ndarray,
    coordinates: pd.DataFrame,
    teachers: dict[str, np.ndarray],
    kind: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    names = list(teachers)
    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    keep = training[rows]
    rows, cols = rows[keep], cols[keep]
    if not len(rows):
        raise ValueError("no masked positives in the model-fitting cells")
    if np.any(counts[rows, cols] != 0):
        raise ValueError("masked coordinates are not zero in the input counts")
    target = np.log1p(coordinates["original_value"].to_numpy(dtype=np.float64)[keep])
    fit_counts = counts[training]
    gene_mean = np.log1p(fit_counts.mean(axis=0)).astype(np.float32)
    detection = (fit_counts > 0).mean(axis=0).astype(np.float32)
    library_log = np.log1p(counts.sum(axis=1)).astype(np.float32)
    teacher_logs = np.column_stack([np.log1p(teachers[name][rows, cols]) for name in names]).astype(np.float32)
    features = value_features(teacher_logs, gene_mean, detection, library_log, rows, cols)

    training_rows = np.flatnonzero(training)
    fold_of_row = np.full(len(counts), -1)
    fold_of_row[training_rows] = training_folds(len(training_rows), seed)
    entry_fold = fold_of_row[rows]
    cross_validation = {}
    for candidate in VALUE_MODELS:
        errors = np.empty(len(target))
        for fold in range(FOLDS):
            held = entry_fold == fold
            fitted = fit_value_model(candidate, teacher_logs[~held], features[~held], target[~held], seed)
            errors[held] = np.abs(np.maximum(predict_value_model(fitted, teacher_logs[held], features[held]), 0.0) - target[held])
        cross_validation[candidate] = {
            "out_of_fold_log1p_mae": float(errors.mean()),
            "fold_log1p_mae": [float(errors[entry_fold == fold].mean()) for fold in range(FOLDS)],
        }

    fitted = fit_value_model(kind, teacher_logs, features, target, seed)
    prediction = np.empty(counts.shape, dtype=np.float32)
    all_cols = np.arange(counts.shape[1])
    for start in range(0, len(counts), 256):
        block = np.arange(start, min(start + 256, len(counts)))
        block_rows = np.repeat(block, len(all_cols))
        block_cols = np.tile(all_cols, len(block))
        block_logs = np.column_stack([np.log1p(teachers[name][block].ravel()) for name in names]).astype(np.float32)
        block_features = value_features(block_logs, gene_mean, detection, library_log, block_rows, block_cols)
        prediction[block] = predict_value_model(fitted, block_logs, block_features).reshape(len(block), -1)
    prediction = np.expm1(np.maximum(prediction, 0.0)).astype(np.float32)
    details = {
        "value_model": kind,
        "value_model_description": (
            "gradient-boosted regression (squared-error loss) of the log1p count on log1p teacher proposals, "
            "their spread, gene mean, gene detection rate and log library size"
            if kind == "boosted" else "least-squares weights of the log1p teacher proposals"
        ),
        "teacher_names": names,
        "value_fit_entries": int(len(rows)),
        "value_fit_cells": "masked positives of model-fitting cells",
        "value_model_cross_validation": cross_validation,
    }
    if kind == "linear":
        details["coefficients"] = {name: float(weight) for name, weight in zip(names, fitted[1][:-1])}
        details["intercept"] = float(fitted[1][-1])
    else:
        details["boosting_iterations"] = int(fitted[1].n_iter_)
    return prediction, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["gene_median", "svd_impute", "graph_smooth", "safe_fusion"],
        required=True,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--value-model", choices=VALUE_MODELS, default="boosted")
    parser.add_argument(
        "--condition-column",
        default=None,
        help="For graph_smooth: borrow only from model-fitting cells with the same value of this obs column.",
    )
    parser.add_argument(
        "--teacher-contract",
        action="append",
        default=[],
        help="Teacher contracts combined by the safe_fusion stack (required for safe_fusion).",
    )
    parser.add_argument(
        "--transductive",
        action="store_true",
        help=(
            "Fit gene_median, svd_impute and graph_smooth on the masked counts of all cells, held-out cells "
            "included, without cross-fitting: each cell's proposal comes from a fit that contains its own "
            "counts, and every cell is in its own kNN neighbour set. For safe_fusion, accept count-scale "
            "teacher contracts fitted this way. The fused value is still fitted on the masked positives of "
            "the model-fitting cells only, and no hidden value is used."
        ),
    )
    args = parser.parse_args()
    if args.method == "safe_fusion" and len(args.teacher_contract) < 2:
        raise ValueError("safe_fusion needs at least two --teacher-contract directories")
    if args.transductive and args.condition_column is not None:
        raise ValueError("--transductive does not support --condition-column")

    adata = ad.read_h5ad(args.input)
    matrix = adata.layers["corrupted_counts"] if "corrupted_counts" in adata.layers else adata.X
    counts = (matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)).astype(np.float32)
    split_frame = pd.read_parquet(args.splits).set_index("cell_id")
    split = split_frame.loc[adata.obs_names.astype(str), "split"].to_numpy()
    training = np.isin(split, ["development", "validation"])
    if not training.any() or np.any(split[training] == "test"):
        raise ValueError("invalid training split")

    teacher_cells = np.ones(len(counts), dtype=bool) if args.transductive else training
    medians = gene_medians(counts, teacher_cells)
    svd = None
    graph = None
    transductive_pca = None
    if args.transductive and args.method in {"svd_impute", "graph_smooth"}:
        transductive_pca = fit_pca(log1p_cpm(counts)[0], args.components, args.seed)
    elif args.method in {"svd_impute", "graph_smooth"}:
        svd = SVDTeacher(counts, training, args.seed, args.components)
    if args.method == "graph_smooth":
        if args.transductive:
            graph = GraphTeacher(counts, teacher_cells, transductive_pca, args.neighbors)
        elif args.condition_column is None:
            assert svd is not None
            graph = GraphTeacher(counts, training, svd.model, args.neighbors)
        else:
            assert svd is not None
            graph = ConditionGraphTeacher(counts, training, adata.obs[args.condition_column].to_numpy(), svd.model, args.neighbors)
    pca_model = transductive_pca if transductive_pca is not None else (svd.model if svd is not None else None)
    details: dict = {
        "fit_cells": int(teacher_cells.sum()) if args.method != "safe_fusion" else int(training.sum()),
        "heldout_test_cells": int((split == "test").sum()),
        "svd_components": int(pca_model.n_components_) if pca_model is not None else None,
        "graph_neighbors": int(args.neighbors) if graph is not None else None,
        "test_used_for_fit": bool(args.transductive),
        "fitting_cell_proposals_exclude_own_counts": not args.transductive,
        "condition_column": args.condition_column,
    }
    if args.transductive:
        details.update({
            "transductive": True,
            "test_labels_used_for_fit": False,
            "cross_fitting": "none: training and held-out cells enter the teachers in the same way",
        })
    if args.method == "gene_median":
        prediction = fill_zeros(counts, medians)
    elif args.method == "svd_impute":
        if args.transductive:
            prediction = svd_reconstruct(transductive_pca, counts)
        else:
            assert svd is not None
            prediction = svd.proposals(counts)
    elif args.method == "graph_smooth":
        assert graph is not None
        if args.transductive:
            prediction = graph.smooth(counts, np.full(len(counts), -1))
        else:
            prediction = graph.proposals(counts)
    else:
        teachers = {}
        for path in args.teacher_contract:
            teacher_metadata = json.loads((Path(path) / "metadata.json").read_text())
            if args.transductive:
                if teacher_metadata.get("scale") != "counts" or not teacher_metadata["parameters"].get("transductive"):
                    raise ValueError(f"teacher {path} must be a count-scale contract fitted on all cells")
            elif teacher_metadata.get("scale") != "counts" or not teacher_metadata["parameters"].get("fitting_cell_proposals_exclude_own_counts"):
                raise ValueError(f"teacher {path} must be a count-scale contract whose fitting-cell proposals exclude own counts")
            if teacher_metadata["cell_ids"] != adata.obs_names.astype(str).tolist():
                raise ValueError(f"teacher {path} cell order differs from the input")
            teachers[Path(path).name] = np.maximum(np.load(Path(path) / "mean.npy"), 0.0).astype(np.float32)
        prediction, value_details = fused_value(counts, training, pd.read_parquet(args.coordinates), teachers, args.value_model, args.seed)
        details.update(value_details)

    try:
        version = subprocess.check_output(
            ["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        version = "unavailable"
    cell_ids = adata.obs_names.astype(str).tolist()
    gene_ids = adata.var_names.astype(str).tolist()
    fitted_cells = training if args.method == "safe_fusion" else teacher_cells
    metadata = {
        "method": args.method,
        "method_version": version,
        "scale": "counts",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": sorted(set(split[fitted_cells].tolist())) if args.transductive else ["development", "validation"],
        "training_data": {"cell_ids_sha256": order_hash(np.asarray(cell_ids)[fitted_cells].tolist())},
        "parameters": details,
        "seed": args.seed,
        "input_sha256": sha256_file(args.input),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, prediction, metadata)
    print(json.dumps({"method": args.method, "shape": list(prediction.shape), "parameters": details}, default=str))


if __name__ == "__main__":
    main()
