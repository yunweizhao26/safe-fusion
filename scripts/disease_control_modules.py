#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings

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
from evaluate_colon_donor_biology import INFLAMMATION_MARKERS
from evaluate_pancreas_crossfit_biology import DISEASE_MARKERS
from safefusion_benchmark.marker_panels import (
    COLON_LABEL_TO_KONG_SUBSET,
    FASOLINO_TABLE_S17,
    KONG_TABLE_S2,
    PANCREAS_TYPE_TO_LABELS,
)


def modules(tissue: str, symbols: set[str]) -> dict[str, tuple[list[str], tuple[str, ...] | None]]:
    if tissue == "pancreas":
        sets = {"disease (interferon)": (sorted(DISEASE_MARKERS), None)}
        sets.update({name: (list(genes), PANCREAS_TYPE_TO_LABELS[name]) for name, genes in FASOLINO_TABLE_S17.items()})
    else:
        sets = {"disease (inflammation)": (sorted(INFLAMMATION_MARKERS), None)}
        sets.update({label: (list(KONG_TABLE_S2[subset]), (label,)) for label, subset in COLON_LABEL_TO_KONG_SUBSET.items()})
    present = {name: ([g for g in genes if g in symbols], target) for name, (genes, target) in sets.items()}
    return {name: value for name, value in present.items() if value[0]}


def score_cells(matrix: np.ndarray, symbols: np.ndarray, sets: dict, seed: int) -> pd.DataFrame:
    adata = ad.AnnData(X=matrix.astype(np.float32))
    adata.var_names = pd.Index(symbols).astype(str)
    adata.var_names_make_unique()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    scores = {}
    for name, (genes, _) in sets.items():
        sc.tl.score_genes(adata, genes, score_name="score", random_state=seed, use_raw=False)
        scores[name] = adata.obs["score"].to_numpy(dtype=np.float64)
    return pd.DataFrame(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    spec = TISSUES[args.tissue]
    output = output_dir("module_scores", args.tissue)

    data = load_heldout(args.tissue)
    cells = data.cells
    sets = modules(args.tissue, set(data.symbols))
    donors = sorted(cells["donor"].unique())
    donor_code = pd.Categorical(cells["donor"], categories=donors).codes
    donor_condition = cells.groupby("donor")["condition_label"].first().loc[donors].to_numpy()
    draws = stratified_draws(donor_condition, args.draws, args.seed)
    labels = cells["cell_type"].to_numpy()

    matrices = {("unfilled", 0.0): data.counts}
    for method in METHODS:
        for fraction in FRACTIONS:
            matrices[(method, fraction)] = filled_matrix(data, method, fraction)[data.heldout]
    scores = {key: score_cells(matrix, data.symbols, sets, args.seed) for key, matrix in matrices.items()}
    pd.concat(
        [frame.assign(method=key[0], fraction=key[1], cell_id=cells["cell_id"].to_numpy()) for key, frame in scores.items()],
        ignore_index=True,
    ).to_parquet(output / "cell_scores.parquet", index=False)

    raw = scores[("unfilled", 0.0)]
    agreement_rows, effect_rows = [], []
    for name, (genes, target_labels) in sets.items():
        target = np.ones(len(cells), dtype=bool) if target_labels is None else np.isin(labels, target_labels)
        count = np.bincount(donor_code[target], minlength=len(donors)).astype(float)

        def donor_mean(values: np.ndarray) -> np.ndarray:
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.bincount(donor_code[target], weights=values[target], minlength=len(donors)) / count

        def contrast(means: np.ndarray, donor_rows: np.ndarray, case: str) -> float:
            conditions = donor_condition[donor_rows]
            values = means[donor_rows]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return float(np.nanmean(values[conditions == case]) - np.nanmean(values[conditions == spec["control"]]))

        raw_means = donor_mean(raw[name].to_numpy())
        weights = [np.bincount(draw, minlength=len(donors))[donor_code] for draw in draws]
        for key, frame in scores.items():
            if key[0] == "unfilled":
                continue
            filled = frame[name].to_numpy()
            change = filled - raw[name].to_numpy()

            def correlation(mask: np.ndarray, weight: np.ndarray | None = None) -> float:
                x, y = raw[name].to_numpy()[mask], filled[mask]
                w = np.ones(mask.sum()) if weight is None else weight[mask]
                if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
                    return float("nan")
                x_centered = x - np.average(x, weights=w)
                y_centered = y - np.average(y, weights=w)
                return float(np.sum(w * x_centered * y_centered) / np.sqrt(np.sum(w * x_centered**2) * np.sum(w * y_centered**2)))

            boot_correlation = [correlation(target & (weight > 0), weight) for weight in weights]
            low, high = interval(np.asarray(boot_correlation))
            other = ~target
            agreement_rows.append({
                "module": name, "n_genes": len(genes), "genes": " ".join(genes),
                "method": key[0], "fraction": key[1], "n_target_cells": int(target.sum()),
                "pearson_target": correlation(target), "pearson_target_ci_low": low, "pearson_target_ci_high": high,
                "mean_change_target": float(change[target].mean()),
                "mean_change_other": float(change[other].mean()) if other.any() else float("nan"),
            })
            filled_means = donor_mean(filled)
            everyone = np.arange(len(donors))
            for case in spec["cases"]:
                observed_raw = contrast(raw_means, everyone, case)
                observed_filled = contrast(filled_means, everyone, case)
                boot_raw = np.asarray([contrast(raw_means, draw, case) for draw in draws])
                boot_filled = np.asarray([contrast(filled_means, draw, case) for draw in draws])
                row = {
                    "module": name, "method": key[0], "fraction": key[1], "contrast": f"{case} vs {spec['control']}",
                    "n_case_donors": int(np.sum((donor_condition == case) & (count > 0))),
                    "n_control_donors": int(np.sum((donor_condition == spec["control"]) & (count > 0))),
                    "effect_unfilled": observed_raw, "effect_filled": observed_filled,
                    "effect_change": observed_filled - observed_raw,
                }
                row["effect_unfilled_ci_low"], row["effect_unfilled_ci_high"] = interval(boot_raw)
                row["effect_filled_ci_low"], row["effect_filled_ci_high"] = interval(boot_filled)
                row["effect_change_ci_low"], row["effect_change_ci_high"] = interval(boot_filled - boot_raw)
                effect_rows.append(row)
    pd.DataFrame(agreement_rows).to_csv(output / "agreement.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(output / "disease_effect.csv", index=False)
    print(pd.DataFrame(effect_rows).query("module.str.startswith('disease')", engine="python").to_string(index=False))


if __name__ == "__main__":
    main()
