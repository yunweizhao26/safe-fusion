#!/usr/bin/env python3
"""Cell-identity macro-F1 from a frozen embedding (e.g., scGPT)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))


def macro_f1(actual: np.ndarray, predicted: np.ndarray) -> float:
    values: list[float] = []
    for label in np.unique(actual):
        tp = np.sum((actual == label) & (predicted == label))
        fp = np.sum((actual != label) & (predicted == label))
        fn = np.sum((actual == label) & (predicted != label))
        denominator = 2 * tp + fp + fn
        values.append(float(2 * tp / denominator) if denominator else 0.0)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.h5ad)
    embeddings = np.load(args.embeddings, allow_pickle=False)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[
        adata.obs_names.astype(str), "split"
    ].to_numpy()
    train = np.isin(split, ["development", "validation"])
    test = split == "test"
    labels = adata.obs["cell_type"].astype(str).to_numpy()
    donors = adata.obs["donor"].astype(str).to_numpy()

    classes = np.unique(labels[train])
    centroids = np.stack([embeddings[train & (labels == label)].mean(axis=0) for label in classes])
    predicted = classes[np.argmin(cdist(embeddings[test], centroids), axis=1)]
    per_donor = {}
    for donor in sorted(set(donors[test])):
        rows = np.flatnonzero(test & (donors == donor))
        local = np.flatnonzero(np.isin(np.flatnonzero(test), rows))
        per_donor[donor] = macro_f1(labels[rows], predicted[local])
    pooled = macro_f1(labels[test], predicted)

    rng = np.random.default_rng(args.seed)
    donor_list = sorted(per_donor)
    values = np.asarray([per_donor[d] for d in donor_list])
    samples = np.asarray([values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(2000)])
    report = {
        "n_test_donors": len(donor_list),
        "pooled_macro_f1": float(pooled),
        "donor_mean_macro_f1": float(values.mean()),
        "donor_bootstrap_ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "per_donor": per_donor,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
