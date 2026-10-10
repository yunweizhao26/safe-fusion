from __future__ import annotations

import numpy as np
import pandas as pd


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average", pct=True).to_numpy()
    return np.minimum((ranks * n_bins).astype(int), n_bins - 1)


def corrupt_counts(
    counts: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    biological_units: np.ndarray,
    splits: np.ndarray,
    spec: dict,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    kind = spec["kind"]
    if kind == "binomial_thinning":
        retained = float(spec["retained_fraction"])
        corrupted = rng.binomial(counts.astype(np.int64), retained).astype(np.int32)
        rows, cols = np.where(corrupted != counts)
    elif kind == "stratified_nonzero_mask":
        corrupted = counts.copy().astype(np.int32)
        rows, cols = np.where(counts > 0)
        gene_abundance = counts.mean(axis=0)
        library_size = counts.sum(axis=1)
        gene_bins = _quantile_bins(gene_abundance, int(spec.get("gene_bins", 4)))
        library_bins = _quantile_bins(library_size, int(spec.get("library_bins", 4)))
        selected: list[int] = []
        strata = np.stack([library_bins[rows], gene_bins[cols]], axis=1)
        for stratum in np.unique(strata, axis=0):
            candidates = np.flatnonzero(np.all(strata == stratum, axis=1))
            n_select = max(1, int(round(len(candidates) * float(spec["fraction"]))))
            selected.extend(rng.choice(candidates, size=min(n_select, len(candidates)), replace=False).tolist())
        selected = sorted(set(selected))
        rows, cols = rows[selected], cols[selected]
        corrupted[rows, cols] = 0
    else:
        raise ValueError(f"Unknown corruption kind: {kind}")
    coordinates = pd.DataFrame({
        "cell_id": cell_ids[rows],
        "gene_id": gene_ids[cols],
        "cell_index": rows.astype(np.int64),
        "gene_index": cols.astype(np.int64),
        "corruption_type": kind,
        "original_value": counts[rows, cols].astype(np.float32),
        "corrupted_value": corrupted[rows, cols].astype(np.float32),
        "biological_unit": biological_units[rows],
        "split": splits[rows],
    })
    if coordinates.duplicated(["cell_id", "gene_id"]).any():
        raise AssertionError("evaluation coordinates are not unique")
    return corrupted, coordinates
