#!/usr/bin/env python3
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from disease_control_common import (
    DRAWS,
    FRACTIONS,
    METHODS,
    SEED,
    TISSUES,
    filled_matrix,
    interval,
    load_heldout,
    output_dir,
    stratified_draws,
)
from evaluate_pancreas_crossfit_biology import centroid_predictions, macro_f1
from safefusion_benchmark.downstream import adjusted_rand_index

RESOLUTION = 1.0
COMPONENTS = 30
NEIGHBORS = 15
SEED_REFERENCE = "unfilled, next seed"


def reference_mapping(data, matrix: np.ndarray, seed: int) -> np.ndarray:
    labels = data.obs["cell_type"].to_numpy()
    assigned = np.empty(len(labels), dtype=object)
    for fold, test in enumerate(data.test_masks.values()):
        fold_matrix = data.recorded.copy()
        fold_matrix[test] = matrix[test]
        fold_seed = seed + fold if data.tissue == "pancreas" else seed
        assigned[test] = centroid_predictions(fold_matrix, labels, ~test, test, fold_seed)
    return assigned[data.heldout].astype(str)


def leiden(matrix: np.ndarray, seed: int) -> np.ndarray:
    adata = ad.AnnData(X=matrix.astype(np.float32))
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=COMPONENTS, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=NEIGHBORS, n_pcs=COMPONENTS, random_state=seed)
    sc.tl.leiden(adata, resolution=RESOLUTION, flavor="igraph", n_iterations=2, directed=False, random_state=seed)
    return adata.obs["leiden"].astype(str).to_numpy()


def mapped_clusters(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    table = pd.crosstab(other, reference)
    return pd.Series(other).map(table.idxmax(axis=1)).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    output = output_dir("annotation", args.tissue)

    data = load_heldout(args.tissue)
    cells = data.cells
    labels = cells["cell_type"].to_numpy()
    donors = sorted(cells["donor"].unique())
    donor_code = pd.Categorical(cells["donor"], categories=donors).codes
    donor_condition = cells.groupby("donor")["condition_label"].first().loc[donors].to_numpy()
    draws = stratified_draws(donor_condition, args.draws, args.seed)
    weights = np.stack([np.bincount(draw, minlength=len(donors))[donor_code] for draw in draws])
    cell_types = sorted(np.unique(labels))

    keys = [("unfilled", 0.0), (SEED_REFERENCE, 0.0)] + [(method, fraction) for method in METHODS for fraction in FRACTIONS]
    assignments, clusters = {}, {}
    for key in keys:
        matrix = data.recorded if key[0] in ("unfilled", SEED_REFERENCE) else filled_matrix(data, *key)
        seed = args.seed + 1 if key[0] == SEED_REFERENCE else args.seed
        assignments[key] = reference_mapping(data, matrix, seed)
        clusters[key] = leiden(matrix[data.heldout], seed)
    pd.concat(
        [pd.DataFrame({"cell_id": cells["cell_id"], "donor": cells["donor"], "cell_type": labels,
                       "method": key[0], "fraction": key[1], "assigned": assignments[key], "leiden": clusters[key]})
         for key in keys],
        ignore_index=True,
    ).to_parquet(output / "cell_assignments.parquet", index=False)

    raw_assigned = assignments[("unfilled", 0.0)]
    raw_clusters = clusters[("unfilled", 0.0)]

    def weighted_share(flags: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        point = float(flags[mask].mean()) if mask.any() else float("nan")
        boot = [float(np.sum(w[mask] * flags[mask]) / np.sum(w[mask])) if np.sum(w[mask]) else float("nan") for w in weights]
        return (point, *interval(np.asarray(boot)))

    def weighted_count(flags: np.ndarray) -> tuple[float, np.ndarray]:
        return float(flags.sum()), weights @ flags.astype(float)

    mapping_rows, overall_rows, leiden_rows, leiden_type_rows = [], [], [], []
    for key in keys[1:]:
        assigned = assignments[key]
        changed = assigned != raw_assigned
        point, low, high = weighted_share(changed, np.ones(len(labels), dtype=bool))
        overall_rows.append({
            "method": key[0], "fraction": key[1], "changed_share": point, "changed_share_ci_low": low, "changed_share_ci_high": high,
            "macro_f1_unfilled": macro_f1(labels, raw_assigned), "macro_f1_filled": macro_f1(labels, assigned),
        })
        for cell_type in cell_types:
            mask = labels == cell_type
            row = {"method": key[0], "fraction": key[1], "cell_type": cell_type, "n_cells": int(mask.sum())}
            row["changed_share"], row["changed_share_ci_low"], row["changed_share_ci_high"] = weighted_share(changed, mask)
            row["recall_unfilled"] = float(np.mean(raw_assigned[mask] == cell_type))
            row["recall_filled"] = float(np.mean(assigned[mask] == cell_type))
            size_raw, boot_raw = weighted_count(raw_assigned == cell_type)
            size_filled, boot_filled = weighted_count(assigned == cell_type)
            row["assigned_unfilled"] = int(size_raw)
            row["assigned_filled"] = int(size_filled)
            row["size_change"] = int(size_filled - size_raw)
            row["size_change_percent"] = 100 * (size_filled - size_raw) / size_raw if size_raw else float("nan")
            with np.errstate(invalid="ignore", divide="ignore"):
                row["size_change_percent_ci_low"], row["size_change_percent_ci_high"] = interval(100 * (boot_filled - boot_raw) / boot_raw)
            mapping_rows.append(row)

        filled_clusters = clusters[key]
        boot_ari = []
        for w in weights:
            repeat = np.repeat(np.arange(len(labels)), w)
            boot_ari.append(adjusted_rand_index(raw_clusters[repeat], filled_clusters[repeat]))
        low, high = interval(np.asarray(boot_ari))
        leiden_rows.append({
            "method": key[0], "fraction": key[1],
            "ari_unfilled_vs_filled": adjusted_rand_index(raw_clusters, filled_clusters),
            "ari_ci_low": low, "ari_ci_high": high,
            "ari_labels_unfilled": adjusted_rand_index(labels, raw_clusters),
            "ari_labels_filled": adjusted_rand_index(labels, filled_clusters),
            "n_clusters_unfilled": int(len(np.unique(raw_clusters))),
            "n_clusters_filled": int(len(np.unique(filled_clusters))),
        })
        moved = mapped_clusters(raw_clusters, filled_clusters) != raw_clusters
        for cell_type in cell_types:
            mask = labels == cell_type
            row = {"method": key[0], "fraction": key[1], "cell_type": cell_type, "n_cells": int(mask.sum())}
            row["changed_share"], row["changed_share_ci_low"], row["changed_share_ci_high"] = weighted_share(moved, mask)
            leiden_type_rows.append(row)

    pd.DataFrame(mapping_rows).to_csv(output / "reference_mapping.csv", index=False)
    pd.DataFrame(overall_rows).to_csv(output / "reference_mapping_overall.csv", index=False)
    pd.DataFrame(leiden_rows).to_csv(output / "leiden.csv", index=False)
    pd.DataFrame(leiden_type_rows).to_csv(output / "leiden_by_cell_type.csv", index=False)
    print(pd.DataFrame(overall_rows).to_string(index=False))
    print(pd.DataFrame(leiden_rows).to_string(index=False))


if __name__ == "__main__":
    main()
