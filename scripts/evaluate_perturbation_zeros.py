#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import roc_auc_score

SCREENS = {
    "norman_crispra": "activation",
    "adamson_crispri": "knockdown",
    "dixit_ko": "knockdown",
    "papalexi_eccite": "knockdown",
}
FRACTIONS = list(range(1, 11))
SPARSE_METHODS = {
    "safe_fusion": "safe_fusion_{pct}pct",
    "safe_fusion_condition": "selector_condition/safe_fusion_calibrated_mlp_topk_{suffix}",
    "svd": "svd_{pct}pct",
    "weighted_knn": "weighted_knn_{pct}pct",
    "magic": "matched_condition/magic_topk_{suffix}",
    "scvi": "matched_condition/scvi_topk_{suffix}",
    "knn_condition": "matched_condition/knn_condition_topk_{suffix}",
    "scvi_condition": "matched_condition/scvi_condition_topk_{suffix}",
}
VALUE_METHODS = ("gene_median", "svd_impute", "graph_smooth", "magic_inductive", "scvi_inductive",
                 "graph_smooth_condition", "scvi_inductive_condition")


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def evaluate_screen(dataset: str, deploy_root: Path, splits_path: Path, prepared_path: Path, min_detection: float) -> pd.DataFrame:
    direction = SCREENS[dataset]
    prepared = ad.read_h5ad(prepared_path)
    hybrid = ad.read_h5ad(deploy_root / "hybrid.h5ad")
    if not np.array_equal(prepared.obs_names.astype(str), hybrid.obs_names.astype(str)):
        raise ValueError(f"{dataset}: prepared and hybrid cell order differ")
    recorded = dense(prepared.layers["counts"]).astype(np.float32)
    symbols = prepared.var["feature_name"].astype(str).to_numpy() if "feature_name" in prepared.var else prepared.var_names.astype(str).to_numpy()
    gene_index = {gene: i for i, gene in enumerate(symbols)}
    split = pd.read_parquet(splits_path).set_index("cell_id").loc[prepared.obs_names.astype(str), "split"].to_numpy()
    target = prepared.obs["target"].astype(str).to_numpy()
    control = target == "none"
    development, test = split == "development", split == "test"
    library = recorded.sum(axis=1, keepdims=True)
    normalized = np.log1p(recorded / np.maximum(library, 1) * 1e4)

    first_fill = {}
    for method, template in SPARSE_METHODS.items():
        if not (deploy_root / template.format(pct=1, suffix="0p01")).exists():
            continue
        order = np.full(recorded.shape, np.inf, dtype=np.float32)
        for pct in reversed(FRACTIONS):
            suffix = "0p1" if pct == 10 else f"0p0{pct}"
            output = np.load(deploy_root / template.format(pct=pct, suffix=suffix) / "mean.npy", mmap_mode="r")
            filled = (recorded == 0) & (np.asarray(output) > 0) & test[:, None]
            order[filled] = pct
        first_fill[method] = order
    values = {name: np.load(deploy_root / "methods" / name / "mean.npy", mmap_mode="r")
              for name in VALUE_METHODS if (deploy_root / "methods" / name / "mean.npy").exists()}

    rows = []
    for gene in sorted(set(target) - {"none"}):
        if gene not in gene_index:
            continue
        g = gene_index[gene]
        dev_control, dev_perturbed = development & control, development & (target == gene)
        expressing = dev_control if direction == "knockdown" else dev_perturbed
        if dev_perturbed.sum() < 5 or (recorded[expressing, g] > 0).mean() < min_detection:
            continue
        test_stat = stats.mannwhitneyu(normalized[dev_perturbed, g], normalized[dev_control, g], alternative="less" if direction == "knockdown" else "greater")
        if test_stat.pvalue >= 0.01:
            continue
        control_zero = np.flatnonzero(test & control & (recorded[:, g] == 0))
        perturbed_zero = np.flatnonzero(test & (target == gene) & (recorded[:, g] == 0))
        if len(control_zero) < 5 or len(perturbed_zero) < 5:
            continue
        cells = np.concatenate([control_zero, perturbed_zero])
        likely_dropout = np.concatenate([np.ones(len(control_zero)), np.zeros(len(perturbed_zero))])
        if direction == "activation":
            likely_dropout = 1 - likely_dropout
        base = dict(dataset=dataset, direction=direction, gene=gene, n_control_zero=len(control_zero), n_perturbed_zero=len(perturbed_zero),
                    development_log2fc=float(np.log2((np.expm1(normalized[dev_perturbed, g]).mean() + 1e-3) / (np.expm1(normalized[dev_control, g]).mean() + 1e-3))))
        for method in first_fill:
            order = first_fill[method][cells, g]
            score = np.where(np.isfinite(order), 11 - order, 0.0)
            auroc = roc_auc_score(likely_dropout, score)
            for pct in FRACTIONS:
                fill_control = float((order[: len(control_zero)] <= pct).mean())
                fill_perturbed = float((order[len(control_zero):] <= pct).mean())
                sign = 1.0 if direction == "knockdown" else -1.0
                rows.append({**base, "method": method, "fill_pct": pct, "fill_control": fill_control, "fill_perturbed": fill_perturbed,
                             "directional_difference": sign * (fill_control - fill_perturbed), "auroc": auroc})
        for method in values:
            score = np.asarray(values[method][cells, g], dtype=np.float64)
            rows.append({**base, "method": f"value:{method}", "fill_pct": np.nan, "fill_control": np.nan, "fill_perturbed": np.nan,
                         "directional_difference": np.nan, "auroc": roc_auc_score(likely_dropout, score)})
    return pd.DataFrame(rows)


def bootstrap(frame: pd.DataFrame, column: str, reference: str, seed: int, draws: int = 2000) -> dict:
    wide = frame.pivot_table(index=["dataset", "gene"], columns="method", values=column)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(wide), size=(draws, len(wide)))
    out = {}
    for method in wide.columns:
        v = wide[method].to_numpy()
        entry = {"mean": float(np.nanmean(v)), "ci": [float(x) for x in np.nanpercentile(np.nanmean(v[idx], axis=1), [2.5, 97.5])]}
        if method != reference:
            d = v - wide[reference].to_numpy()
            entry[f"{reference}_minus_method"] = [float(-np.nanmean(d))] + [float(-x) for x in np.nanpercentile(np.nanmean(d[idx], axis=1), [97.5, 2.5])]
        out[method] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy-root", default="artifacts/paper_evidence/downstream_deployment")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence/perturbation_zeros")
    parser.add_argument("--min-detection", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    splits = {
        "norman_crispra": Path("artifacts/paper_evidence/norman_crispra/splits.parquet"),
        "adamson_crispri": Path("artifacts/external_perturbseq/adamson_crispri/splits.parquet"),
        "dixit_ko": Path("artifacts/external_perturbseq/dixit_ko/splits.parquet"),
        "papalexi_eccite": Path("artifacts/external_perturbseq/papalexi_eccite/splits.parquet"),
    }
    frames = [evaluate_screen(ds, Path(args.deploy_root) / ds, splits[ds], Path(f"external_data/prepared/{ds}.h5ad"), args.min_detection) for ds in SCREENS]
    frame = pd.concat(frames, ignore_index=True)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "per_target.csv", index=False)
    report = {"targets": frame.groupby("dataset")["gene"].nunique().to_dict(), "min_control_detection": args.min_detection}
    ranking = frame.drop_duplicates(["dataset", "gene", "method"])
    for scope, sub in [("knockdown", ranking[ranking.direction == "knockdown"]), ("activation", ranking[ranking.direction == "activation"])]:
        if len(sub):
            report[f"auroc_{scope}"] = bootstrap(sub, "auroc", "safe_fusion", args.seed)
    report["auroc_by_screen"] = ranking.groupby(["dataset", "method"])["auroc"].mean().round(4).unstack().to_dict(orient="index")
    fills = frame[frame.method.isin(list(SPARSE_METHODS))]
    for scope in ["knockdown", "activation"]:
        report[f"fill_{scope}"] = {}
        for pct in [1, 2, 5, 10]:
            sub = fills[(fills.direction == scope) & (fills.fill_pct == pct)]
            if not len(sub):
                continue
            report[f"fill_{scope}"][pct] = {
                "directional_difference": bootstrap(sub, "directional_difference", "safe_fusion", args.seed),
                "fill_control": sub.groupby("method")["fill_control"].mean().round(4).to_dict(),
                "fill_perturbed": sub.groupby("method")["fill_perturbed"].mean().round(4).to_dict(),
            }
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1)[:6000])


if __name__ == "__main__":
    main()
