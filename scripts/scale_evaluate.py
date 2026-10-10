#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from compute_matched_baseline_f1_curves import tie_broken_order
from masked_f1_units import COUNT_SCALE, UNIT_FRACTIONS

SELECTORS = {
    "Safe Fusion": "selector",
    "Safe Fusion (transductive)": "transductive/selector",
    "Safe Fusion without MAGIC teacher": "without_magic/selector",
}
METHODS = {
    "SVD": "methods/svd_impute",
    "Weighted kNN": "methods/graph_smooth",
    "ALRA": "baselines/alra",
    "SAVER": "baselines/saver",
    "MAGIC": "baselines/magic",
    "scVI": "baselines/scvi",
    "scVI probability": "baselines/scvi_probability",
    "MAGIC (inductive)": "methods/magic_inductive",
    "scVI (inductive)": "methods/scvi_inductive",
}
RANKED_AS_STORED = {"negative_log_zero_probability"}
DRAW_CHUNK = 25

def method_scores(contract: Path, rows: np.ndarray, cols: np.ndarray, library: np.ndarray, cell_ids: list[str],
                  batch: int) -> np.ndarray:
    metadata = json.loads((contract / "metadata.json").read_text())
    if metadata["cell_ids"] != cell_ids:
        raise ValueError(f"cell order of {contract} differs from the masked input")
    scale = metadata["scale"]
    mean = np.load(contract / "mean.npy", mmap_mode="r")
    scores = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), batch):
        r, c = rows[start:start + batch], cols[start:start + batch]
        values = np.asarray(mean[r, c], dtype=np.float64)
        scores[start:start + batch] = values if scale in RANKED_AS_STORED else COUNT_SCALE[scale](values, library[r])
    return scores

class PooledRanking:

    def __init__(self, order: np.ndarray, labels: np.ndarray, unit: np.ndarray, n_units: int):
        n = len(order)
        self.selected = np.empty((n_units, len(UNIT_FRACTIONS)))
        self.true_positive = np.empty_like(self.selected)
        self.positives = np.bincount(unit[labels == 1], minlength=n_units).astype(np.float64)
        self.zeros = np.bincount(unit, minlength=n_units)
        for column, fraction in enumerate(UNIT_FRACTIONS):
            top = order[:max(1, int(round(fraction * n)))]
            self.selected[:, column] = np.bincount(unit[top], minlength=n_units)
            self.true_positive[:, column] = np.bincount(unit[top], weights=labels[top], minlength=n_units)
        rank = np.empty(n, dtype=np.int64)
        rank[order] = np.arange(n)
        positive = np.flatnonzero(labels == 1)
        positive = positive[np.argsort(rank[positive])]
        positive_rank = rank[positive]
        self.positive_unit = unit[positive]
        self.cumulative_true = np.zeros((len(positive), n_units), dtype=np.float32)
        self.cumulative_selected = np.zeros((len(positive), n_units), dtype=np.float32)
        for index in range(n_units):
            self.cumulative_true[:, index] = np.cumsum(self.positive_unit == index)
            self.cumulative_selected[:, index] = np.searchsorted(np.sort(rank[unit == index]), positive_rank, side="right")

    def average_precision(self, weights: np.ndarray) -> np.ndarray:

        result = np.empty(len(weights))
        for start in range(0, len(weights), DRAW_CHUNK):
            chunk = weights[start:start + DRAW_CHUNK].T.astype(np.float32)
            true_so_far, selected_so_far = self.cumulative_true @ chunk, self.cumulative_selected @ chunk
            precision = np.divide(true_so_far, selected_so_far, out=np.zeros_like(true_so_far), where=selected_so_far > 0)
            numerator = (chunk[self.positive_unit] * precision).sum(axis=0, dtype=np.float64)
            result[start:start + DRAW_CHUNK] = numerator / (self.positives @ chunk)
        return result

    def mean_f1(self, weights: np.ndarray) -> np.ndarray:
        selected, true_positive = weights @ self.selected, weights @ self.true_positive
        return (2.0 * true_positive / (selected + (weights @ self.positives)[:, None])).mean(axis=1)

def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/paper_evidence/review_round3/scale"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[25_000, 50_000, 100_000, 200_000])
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--batch-rows", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    unit_rows, curve_rows, summary_rows, difference_rows, availability = [], [], [], [], []
    for size in args.sizes:
        root = args.root / f"cells_{size}"
        if not (root / "selector" / "test_score.npy").exists():
            availability.append({"cells": size, "method": "Safe Fusion", "available": False})
            continue
        dataset, group = f"{size} cells", f"cells_{size}"
        adata = ad.read_h5ad(root / "masked" / "corrupted.h5ad")
        cell_ids = adata.obs_names.astype(str).tolist()
        donors = adata.obs["donor"].astype(str).to_numpy()
        library = np.asarray(sparse.csr_matrix(adata.layers["corrupted_counts"]).sum(axis=1, dtype=np.float64)).ravel()
        del adata
        with np.load(root / "selector" / "test_candidates.npz") as data:
            rows, cols, labels = data["rows"], data["cols"], data["labels"].astype(np.int64)
        units = np.asarray(sorted(set(donors[rows])))
        unit = np.searchsorted(units, donors[rows])
        draws = np.random.default_rng(args.seed).integers(0, len(units), size=(args.draws, len(units)))
        weights = np.stack([np.bincount(draw, minlength=len(units)) for draw in draws]).astype(np.float64)
        pooled = {}
        for method, relative in SELECTORS.items():
            directory = root / relative
            available = (directory / "test_score.npy").exists()
            availability.append({"cells": size, "method": method, "available": available})
            if available:
                with np.load(directory / "test_candidates.npz") as data:
                    if not (np.array_equal(data["rows"], rows) and np.array_equal(data["cols"], cols)):
                        raise ValueError(f"{directory} scores other candidates than {root / 'selector'}")
                order = np.argsort(-np.load(directory / "test_score.npy"), kind="stable")
                pooled[method] = PooledRanking(order, labels, unit, len(units))
        for method, relative in METHODS.items():
            contract = root / relative
            available = (contract / "mean.npy").exists()
            availability.append({"cells": size, "method": method, "available": available})
            if available:
                scores = method_scores(contract, rows, cols, library, cell_ids, args.batch_rows)
                pooled[method] = PooledRanking(tie_broken_order(scores, args.seed), labels, unit, len(units))
        ones = np.ones((1, len(units)))
        statistics = {}
        for method, ranking in pooled.items():
            for column, fraction in enumerate(UNIT_FRACTIONS):
                unit_rows.append(pd.DataFrame({
                    "dataset": dataset, "group": group, "method": method, "fraction": fraction, "unit": units,
                    "n_selected": ranking.selected[:, column].astype(np.int64),
                    "n_true_positive": ranking.true_positive[:, column].astype(np.int64),
                    "n_masked_positives": ranking.positives.astype(np.int64), "n_zeros": ranking.zeros,
                }))
                selected, true_positive = ranking.selected[:, column].sum(), ranking.true_positive[:, column].sum()
                curve_rows.append({"cells": size, "method": method, "fraction": fraction,
                                   "masked_f1": 2.0 * true_positive / (selected + ranking.positives.sum())})
            statistics[method] = {
                "mean_f1_1_to_10": (float(ranking.mean_f1(ones)[0]), ranking.mean_f1(weights)),
                "average_precision": (float(ranking.average_precision(ones)[0]), ranking.average_precision(weights)),
            }
            for name, (estimate, boot) in statistics[method].items():
                summary_rows.append({"cells": size, "method": method, "statistic": name, "estimate": estimate,
                                     "lower": interval(boot)[0], "upper": interval(boot)[1],
                                     "test_donors": int(len(units)), "test_candidates": int(len(labels)),
                                     "masked_positives": int(labels.sum())})
        for reference in [name for name in SELECTORS if name in statistics]:
            for method in statistics:
                if method == reference:
                    continue
                for name, (estimate, boot) in statistics[reference].items():
                    other_estimate, other_boot = statistics[method][name]
                    lower, upper = interval(boot - other_boot)
                    difference_rows.append({"cells": size, "reference": reference, "method": method, "statistic": name,
                                            "difference": estimate - other_estimate, "lower": lower, "upper": upper,
                                            "n_units": int(len(units))})
        print(pd.DataFrame([row for row in summary_rows if row["cells"] == size]).to_string(index=False))

    output = args.root / "results"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(unit_rows, ignore_index=True).to_parquet(output / "masked_f1_unit_counts.parquet", index=False)
    pd.DataFrame(curve_rows).to_csv(output / "masked_f1_curves.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "method_summary.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(output / "paired_differences.csv", index=False)
    pd.DataFrame(availability).to_csv(output / "method_availability.csv", index=False)

if __name__ == "__main__":
    main()
