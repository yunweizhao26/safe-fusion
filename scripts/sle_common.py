from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compute_matched_baseline_f1_curves import tie_broken_order
from masked_f1_units import COUNT_SCALE

CASE = ROOT / "artifacts" / "paper_evidence" / "sle_treg_case"
CONFIG = ROOT / "configs" / "sle_treg_genes.json"
FOLDS = 3
LEVELS = np.arange(1, 11)
REPORTED = (1, 5, 10)
NOT_FILLED = 255
SEED = 1729
METHODS = {
    "Safe Fusion": None,
    "SVD": "methods/svd_impute",
    "Weighted kNN": "methods/graph_smooth",
    "MAGIC": "baselines/magic",
    "scVI": "baselines/scvi",
}

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def config() -> dict:
    return json.loads(CONFIG.read_text())

@dataclass
class Unit:
    set_name: str
    fold: int
    app: str
    counts: np.ndarray
    recorded: np.ndarray
    obs: pd.DataFrame
    genes: list[str]
    test: np.ndarray
    masked: np.ndarray | None

    @property
    def directory(self) -> Path:
        return CASE / self.set_name / f"fold_{self.fold}" / self.app

def load_unit(set_name: str, fold: int, app: str, case: Path = CASE) -> Unit:
    fold_dir = case / set_name / f"fold_{fold}"
    model_input = fold_dir / app / ("corrupted.h5ad" if app == "masked" else "hybrid.h5ad")
    adata = ad.read_h5ad(model_input)
    truth = ad.read_h5ad(fold_dir / "truth.h5ad")
    if not np.array_equal(adata.obs_names, truth.obs_names) or not np.array_equal(adata.var_names, truth.var_names):
        raise ValueError(f"cell or gene order differs in {fold_dir}")
    split = pd.read_parquet(fold_dir / "splits.parquet").set_index("cell_id").loc[adata.obs_names, "split"].to_numpy()
    masked = None
    if app == "masked":
        coordinates = pd.read_parquet(fold_dir / app / "coordinates.parquet")
        masked = np.zeros(adata.shape, dtype=bool)
        masked[coordinates["cell_index"].to_numpy(int), coordinates["gene_index"].to_numpy(int)] = True
    return Unit(
        set_name=set_name, fold=fold, app=app,
        counts=dense(adata.layers["corrupted_counts"]).astype(np.float32),
        recorded=dense(truth.layers["counts"]).astype(np.float32),
        obs=truth.obs.copy(), genes=truth.var_names.astype(str).tolist(), test=split == "test", masked=masked,
    )

def candidates(unit: Unit) -> tuple[np.ndarray, np.ndarray]:
    return np.where((unit.counts == 0) & unit.test[:, None])

def selector_scores(unit: Unit, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    table = pq.read_table(
        unit.directory / "selector" / "selected_gene_scores.parquet",
        columns=["split", "cell_index", "gene_index", "selector_score"], filters=[("split", "=", "test")],
    ).to_pandas()
    order = np.lexsort((table["gene_index"].to_numpy(), table["cell_index"].to_numpy()))
    table = table.iloc[order]
    if not (np.array_equal(table["cell_index"].to_numpy(), rows) and np.array_equal(table["gene_index"].to_numpy(), cols)):
        raise ValueError(f"selector scores do not cover the candidates of {unit.directory}")
    return table["selector_score"].to_numpy(np.float32)

def method_values(unit: Unit, method: str, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    if method == "Safe Fusion":
        value = np.load(unit.directory / "methods" / "safe_fusion" / "mean.npy", mmap_mode="r")[rows, cols]
        return selector_scores(unit, rows, cols), np.asarray(value, dtype=np.float32)
    contract = unit.directory / METHODS[method]
    metadata = json.loads((contract / "metadata.json").read_text())
    if metadata["cell_ids"] != unit.obs.index.astype(str).tolist() or metadata["gene_ids"] != unit.genes:
        raise ValueError(f"cell or gene order differs for {contract}")
    library = unit.counts.sum(axis=1, dtype=np.float64)
    raw = np.asarray(np.load(contract / "mean.npy", mmap_mode="r")[rows, cols], dtype=np.float64)
    value = np.maximum(COUNT_SCALE[metadata["scale"]](raw, library[rows]), 0.0).astype(np.float32)
    return value, value

def first_fill(order: np.ndarray, groups: np.ndarray | None) -> np.ndarray:

    n = len(order)
    rank = np.empty(n, dtype=np.int64)
    if groups is None:
        rank[order] = np.arange(n)
        sizes = np.full(n, n)
    else:
        ordered_groups = groups[order]
        position = np.argsort(ordered_groups, kind="stable")
        group_sorted = ordered_groups[position]
        starts = np.searchsorted(group_sorted, group_sorted, side="left")
        within = np.arange(n) - starts
        rank[order[position]] = within
        sizes = np.bincount(groups)[groups]
    first = np.full(n, NOT_FILLED, dtype=np.uint8)
    for level in LEVELS[::-1]:
        k = np.maximum(1, np.round(level / 100 * sizes).astype(np.int64))
        first[rank < k] = level
    return first

def fill_path(unit: Unit, method: str) -> Path:
    return unit.directory / "fills" / f"{method.replace(' ', '_')}.npz"

def load_fills(unit: Unit, method: str) -> dict[str, np.ndarray]:
    with np.load(fill_path(unit, method)) as data:
        return {key: data[key] for key in data.files}

def filled_matrix(unit: Unit, fills: dict[str, np.ndarray], level: int, mode: str) -> np.ndarray:

    test_rows = np.flatnonzero(unit.test)
    output = unit.counts[test_rows].copy()
    selected = fills[f"first_{mode}"] <= level
    local = np.searchsorted(test_rows, fills["rows"][selected])
    output[local, fills["cols"][selected]] = fills["value"][selected]
    return output

def bootstrap_draws(units: np.ndarray, strata: np.ndarray | None, draws: int = 2000, seed: int = SEED) -> np.ndarray:

    rng = np.random.default_rng(seed)
    weights = np.zeros((draws, len(units)))
    groups = [np.arange(len(units))] if strata is None else [np.flatnonzero(strata == value) for value in np.unique(strata)]
    for members in groups:
        picks = members[rng.integers(0, len(members), size=(draws, len(members)))]
        for index in range(draws):
            np.add.at(weights[index], picks[index], 1.0)
    return weights

def interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

def pair_auroc(pos_scores: list[np.ndarray], neg_scores: list[np.ndarray]) -> np.ndarray:

    table = np.zeros((len(pos_scores), len(neg_scores)))
    for b, negatives in enumerate(neg_scores):
        ordered = np.sort(negatives)
        for a, positives in enumerate(pos_scores):
            if len(positives) and len(ordered):
                below = np.searchsorted(ordered, positives, side="left")
                ties = np.searchsorted(ordered, positives, side="right") - below
                table[a, b] = below.sum() + 0.5 * ties.sum()
    return table

def weighted_auroc(u: np.ndarray, n_pos: np.ndarray, n_neg: np.ndarray, w_pos: np.ndarray, w_neg: np.ndarray) -> np.ndarray:

    numerator = np.einsum("ka,ab,kb->k", w_pos, u, w_neg)
    denominator = (w_pos @ n_pos) * (w_neg @ n_neg)
    return np.divide(numerator, denominator, out=np.full(len(w_pos), np.nan), where=denominator > 0)
