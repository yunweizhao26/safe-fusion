#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats

UNFILLED = "unfilled"
PERCENTILES = (50, 95, 99)

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def parse_pair(value: str) -> tuple[str, str]:
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("arguments must use name=value")
    return name, path

def method_parts(name: str) -> tuple[str, float]:
    match = re.fullmatch(r"(.+)_(\d+)pct", name)
    if name == UNFILLED or match is None:
        return name, 0.0
    return match.group(1), int(match.group(2)) / 100.0

def spearman_upper(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    n = values.shape[0]
    ranks = stats.rankdata(values, axis=0).astype(np.float64)
    ranks -= ranks.mean(axis=0)
    norms = np.sqrt(np.sum(ranks * ranks, axis=0))
    constant = norms == 0
    norms[constant] = 1.0
    ranks /= norms
    rho = ranks.T @ ranks
    rho[constant, :] = 0.0
    rho[:, constant] = 0.0
    upper = np.triu_indices(values.shape[1], k=1)
    rho = np.clip(rho[upper], -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = rho * np.sqrt((n - 2) / np.maximum(1.0 - rho * rho, 0.0))
    pvalue = 2.0 * stats.t.sf(np.abs(t), n - 2)
    pvalue = np.nan_to_num(pvalue, nan=1.0)
    return rho, pvalue

def log_cp10k(matrix: np.ndarray) -> np.ndarray:
    library = matrix.sum(axis=1, dtype=np.float64)
    factor = np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)
    return np.log1p(matrix * factor[:, None])

def load_matrix(path: Path, cell_ids: list[str], gene_ids: list[str]) -> np.ndarray:
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
        raise ValueError(f"cell or gene order differs for {path}")
    return np.load(path / "mean.npy", mmap_mode="r")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", action="append", required=True,
                        help="key=unit_directory holding recorded.h5ad (unfilled counts) and splits.parquet; "
                             "the key's first path component is the tissue")
    parser.add_argument("--method", action="append", default=[],
                        help="name=contract template; {unit} is the unit directory and {key} the unit key; "
                             "names end in _<percent>pct")
    parser.add_argument("--label-column", default="cell_type")
    parser.add_argument("--scale", choices=["counts", "log_cp10k"], default="counts")
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=2000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--reference-method", default="safe_fusion",
                        help="method whose paired differences from every other method at the same fraction are written")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    units = [parse_pair(value) for value in args.unit]
    methods = [parse_pair(value) for value in args.method]
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    group_rows: list[dict] = []
    histograms: dict[str, dict[str, np.ndarray]] = {}
    absolute: dict[tuple[str, str], list[np.ndarray]] = {}
    inputs: dict[str, dict] = {}

    for key, unit_dir in units:
        unit = Path(unit_dir)
        tissue = key.split("/")[0]
        adata = ad.read_h5ad(unit / "recorded.h5ad")
        unfilled = dense(adata.layers["corrupted_counts"]).astype(np.float32)
        cell_ids = adata.obs_names.astype(str).tolist()
        gene_ids = adata.var_names.astype(str).tolist()
        split = pd.read_parquet(unit / "splits.parquet").set_index("cell_id").loc[cell_ids, "split"].to_numpy()
        test = split == "test"
        labels = adata.obs[args.label_column].astype(str).to_numpy()
        matrices = {UNFILLED: unfilled}
        paths = {}
        for name, template in methods:
            path = Path(template.format(unit=unit_dir, key=key))
            matrices[name] = load_matrix(path, cell_ids, gene_ids)
            paths[name] = str(path)
            test_values = np.asarray(matrices[name][test])
            recorded_nonzero = unfilled[test] > 0
            if not np.array_equal(test_values[recorded_nonzero], unfilled[test][recorded_nonzero]):
                raise ValueError(f"{path} changes recorded nonzero counts of held-out cells")
        inputs[key] = {"unit_dir": unit_dir, "tissue": tissue, "methods": paths}
        for level in sorted(np.unique(labels[test])):
            rows = np.flatnonzero(test & (labels == level))
            genes = np.flatnonzero((unfilled[rows] > 0).any(axis=0))
            group = f"{key}|{level}"
            histograms[group] = {}
            for name, matrix in matrices.items():
                values = np.asarray(matrix[rows], dtype=np.float64)
                if args.scale == "log_cp10k":
                    values = log_cp10k(values)
                values = values[:, genes]
                rho, pvalue = spearman_upper(values)
                correlated = stats.false_discovery_control(pvalue) < args.fdr
                magnitude = np.abs(rho)
                histograms[group][name] = np.histogram(magnitude, bins=edges)[0].astype(np.int64)
                absolute.setdefault((tissue, name), []).append(magnitude.astype(np.float32))
                changed = int(np.sum(np.asarray(matrix[rows]) != unfilled[rows]))
                method, fraction = method_parts(name)
                group_rows.append({
                    "group": group, "unit": key, "tissue": tissue, "cell_type": level,
                    "matrix": name, "method": method, "fraction": fraction,
                    "n_cells": int(len(rows)), "n_genes": int(len(genes)), "n_pairs": int(len(rho)),
                    "n_correlated": int(correlated.sum()),
                    "n_correlated_positive": int((correlated & (rho > 0)).sum()),
                    "sum_abs_rho": float(magnitude.sum()),
                    "n_filled_entries": changed,
                })
            print(json.dumps({"group": group, "n_cells": int(len(rows)), "n_genes": int(len(genes))}), flush=True)

    frame = pd.DataFrame(group_rows)
    names = [UNFILLED] + [name for name, _ in methods]
    summary_rows: list[dict] = []
    versus_rows: list[dict] = []
    centers = 0.5 * (edges[:-1] + edges[1:])

    def percentile_from_histogram(counts: np.ndarray, q: float) -> float:
        cumulative = np.cumsum(counts)
        if cumulative[-1] == 0:
            return float("nan")
        return float(centers[np.searchsorted(cumulative, q / 100.0 * cumulative[-1])])

    for scope in ["pooled"] + sorted(frame["tissue"].unique()):
        scoped = frame if scope == "pooled" else frame[frame["tissue"] == scope]
        groups = sorted(scoped["group"].unique())
        rng = np.random.default_rng(args.seed)
        draws = rng.integers(0, len(groups), size=(args.bootstrap, len(groups)))
        table = {
            column: scoped.pivot(index="group", columns="matrix", values=column).reindex(index=groups, columns=names)
            for column in ("n_correlated", "n_pairs", "sum_abs_rho")
        }
        hist = {name: np.stack([histograms[group][name] for group in groups]) for name in names}
        boot: dict[str, dict[str, np.ndarray]] = {}
        point: dict[str, dict[str, float]] = {}
        for name in names:
            correlated = table["n_correlated"][name].to_numpy(dtype=float)
            pairs = table["n_pairs"][name].to_numpy(dtype=float)
            sum_abs = table["sum_abs_rho"][name].to_numpy(dtype=float)
            tissues = sorted(scoped["tissue"].unique())
            pooled_abs = np.concatenate([part for tissue in tissues for part in absolute[(tissue, name)]])
            point[name] = {
                "n_correlated": float(correlated.sum()),
                "share_correlated": float(correlated.sum() / pairs.sum()),
                "mean_abs_rho": float(sum_abs.sum() / pairs.sum()),
                **{f"p{q}_abs_rho": float(np.percentile(pooled_abs, q)) for q in PERCENTILES},
            }
            sampled_hist = hist[name][draws].sum(axis=1)
            boot[name] = {
                "n_correlated": correlated[draws].sum(axis=1),
                "share_correlated": correlated[draws].sum(axis=1) / pairs[draws].sum(axis=1),
                "mean_abs_rho": sum_abs[draws].sum(axis=1) / pairs[draws].sum(axis=1),
                **{f"p{q}_abs_rho": np.asarray([percentile_from_histogram(row, q) for row in sampled_hist]) for q in PERCENTILES},
            }
        for name in names:
            method, fraction = method_parts(name)
            record = {
                "scope": scope, "matrix": name, "method": method, "fraction": fraction,
                "n_groups": len(groups), "n_pairs": int(table["n_pairs"][name].sum()),
            }
            for statistic, value in point[name].items():
                samples = boot[name][statistic]
                record[statistic] = value
                record[f"{statistic}_low"] = float(np.nanquantile(samples, 0.025))
                record[f"{statistic}_high"] = float(np.nanquantile(samples, 0.975))
                if name != UNFILLED:
                    difference = samples - boot[UNFILLED][statistic]
                    record[f"{statistic}_difference"] = value - point[UNFILLED][statistic]
                    record[f"{statistic}_difference_low"] = float(np.nanquantile(difference, 0.025))
                    record[f"{statistic}_difference_high"] = float(np.nanquantile(difference, 0.975))
            summary_rows.append(record)
        for name in names:
            method, fraction = method_parts(name)
            reference = f"{args.reference_method}_{int(round(100 * fraction))}pct"
            if name in (UNFILLED, reference) or reference not in point:
                continue
            record = {"scope": scope, "reference": reference, "matrix": name, "method": method, "fraction": fraction}
            for statistic in ("n_correlated", "mean_abs_rho"):
                difference = boot[reference][statistic] - boot[name][statistic]
                record[f"{statistic}_difference"] = point[reference][statistic] - point[name][statistic]
                record[f"{statistic}_difference_low"] = float(np.nanquantile(difference, 0.025))
                record[f"{statistic}_difference_high"] = float(np.nanquantile(difference, 0.975))
            versus_rows.append(record)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "group_statistics.csv", index=False)
    np.savez_compressed(
        output / "histograms.npz",
        groups=np.asarray(sorted(histograms)),
        matrices=np.asarray(names),
        edges=edges,
        counts=np.stack([np.stack([histograms[group][name] for name in names]) for group in sorted(histograms)]),
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(versus_rows).to_csv(output / "versus_reference.csv", index=False)
    report = {
        "design": "gene-gene Spearman correlation within each held-out cell type of each unit, genes with a nonzero unfilled count in the group",
        "scale": args.scale,
        "fdr": args.fdr,
        "bootstrap_unit": "held-out cell type within unit",
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
        "histogram_bins": args.bins,
        "n_groups": int(frame["group"].nunique()),
        "inputs": inputs,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    columns = ["scope", "matrix", "n_correlated", "n_correlated_low", "n_correlated_high", "mean_abs_rho", "p99_abs_rho"]
    print(summary[columns].to_string(index=False))

if __name__ == "__main__":
    main()
