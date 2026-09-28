#!/usr/bin/env python3
"""Compare which recorded zeros Safe Fusion, SVD, and weighted kNN fill.

Tissues: the matched outputs at fill fractions 1% to 10% are nested, so each
candidate zero gets the smallest fraction at which a method fills it. The
filled sets are compared by overlap, masked precision, gene detection rate,
and off-target marker fill for each marker gene.

Papalexi CD274: the selector score is compared with the teacher values among
held-out cells with recorded zero CD274 RNA, including partial Spearman
correlations with surface PD-L1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from safefusion_benchmark.marker_panels import PANELS  # noqa: E402

FRACTIONS = range(1, 11)
COLON = "artifacts/colon_runs/0b2469810675-c0db6f963e94"
PANCREAS = "artifacts/pancreas_runs/0b2469810675-45c81b160d78"
CROSSFIT = "artifacts/paper_evidence/pancreas_crossfit"
PAPALEXI = "artifacts/paper_evidence/papalexi_crossmodal/benchmark"


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def suffix(pct: int) -> str:
    return f"{pct / 100:g}".replace(".", "p")


def fill_level(directory, rows: np.ndarray, cols: np.ndarray, cell_ids: list[str]) -> np.ndarray:
    """Smallest fill fraction in percent at which each candidate is filled, 11 if never."""
    level = np.full(len(rows), 11, dtype=np.int8)
    for pct in sorted(FRACTIONS, reverse=True):
        path = Path(directory(pct))
        if json.loads((path / "metadata.json").read_text())["cell_ids"] != cell_ids:
            raise ValueError(f"cell order differs in {path}")
        level[np.load(path / "mean.npy")[rows, cols] != 0] = pct
    return level


def compare_tissue(name, truth, corrupted, labels, split, genes, targets, methods, cell_ids):
    test = split == "test"
    rows, cols = np.where((corrupted == 0) & test[:, None])
    masked = truth[rows, cols] > 0
    detection = (corrupted[~test] > 0).mean(axis=0)
    levels = {method: fill_level(directory, rows, cols, cell_ids) for method, directory in methods.items()}

    lookup = {gene: index for index, gene in enumerate(genes)}
    target_rows = {}
    for gene, rule in targets.items():
        if gene in lookup:
            target = rule(labels)
            if target.any() and not target.all():
                target_rows[gene] = target
    off_target = np.zeros(len(rows), dtype=bool)
    for gene, target in target_rows.items():
        at_gene = cols == lookup[gene]
        off_target[at_gene] = ~target[rows[at_gene]]

    summary = []
    names = list(levels)
    for pct in (1, 5, 10):
        filled = {method: level <= pct for method, level in levels.items()}
        record = {"dataset": name, "fill_pct": pct, "masked_prevalence": masked.mean()}
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                shared = (filled[first] & filled[second]).sum()
                record[f"overlap_{first}_{second}"] = shared / min(filled[first].sum(), filled[second].sum())
        for method, chosen in filled.items():
            record[f"{method}_masked_precision"] = masked[chosen].mean()
            record[f"{method}_gene_detection"] = detection[cols[chosen]].mean()
        summary.append(record)

    per_marker = []
    for gene, target in target_rows.items():
        entries = (cols == lookup[gene]) & off_target
        zeros = entries & ~masked
        record = {
            "dataset": name,
            "marker": gene,
            "nontarget_cells_detected": (truth[test & ~target, lookup[gene]] > 0).mean(),
            "masked_share_of_offtarget_zeros": masked[entries].mean(),
            "n_offtarget_zeros": int(zeros.sum()),
        }
        for method, level in levels.items():
            record[f"{method}_offtarget_fill_10pct"] = (level[zeros] <= 10).mean()
        per_marker.append(record)
    return pd.DataFrame(summary), pd.DataFrame(per_marker)


def tissues(output: Path) -> None:
    summaries, markers = [], []
    truth_adata = ad.read_h5ad(f"{COLON}/data/colon_epithelial/preprocessed.h5ad")
    corrupted_adata = ad.read_h5ad(f"{COLON}/data/colon_epithelial/corrupted/mask_010.h5ad")
    cell_ids = corrupted_adata.obs_names.astype(str).tolist()
    split = pd.read_parquet(f"{COLON}/data/colon_epithelial/splits.parquet").set_index("cell_id").loc[cell_ids, "split"].to_numpy()
    selector = "artifacts/paper_evidence/selector_mlp_biology_range/colon"
    matched = "artifacts/paper_evidence/matched_fraction/colon"
    result = compare_tissue(
        "colon",
        dense(truth_adata.layers["counts"]).astype(np.float32),
        dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32),
        truth_adata.obs["cell_type"].astype(str).to_numpy(),
        split,
        truth_adata.var["feature_name"].astype(str).tolist(),
        {gene: (lambda labels, cell_types=cell_types: np.isin(labels, np.asarray(cell_types, dtype=str))) for gene, cell_types in PANELS["colon"]["source"].items()},
        {
            "safe_fusion": lambda pct: f"{selector}/safe_fusion_calibrated_mlp_topk_{suffix(pct)}",
            "svd": lambda pct: f"{matched}/svd_topk_{suffix(pct)}",
            "weighted_knn": lambda pct: f"{matched}/weighted_knn_topk_{suffix(pct)}",
        },
        cell_ids,
    )
    summaries.append(result[0])
    markers.append(result[1])

    truth_adata = ad.read_h5ad(f"{PANCREAS}/data/pancreas_islets/preprocessed.h5ad")
    corrupted_adata = ad.read_h5ad(f"{PANCREAS}/data/pancreas_islets/corrupted/mask_010.h5ad")
    cell_ids = corrupted_adata.obs_names.astype(str).tolist()
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    for fold in range(3):
        base = f"{CROSSFIT}/fold_{fold}"
        split = pd.read_parquet(f"{base}/splits.parquet").set_index("cell_id").loc[cell_ids, "split"].to_numpy()
        result = compare_tissue(
            f"pancreas_fold_{fold}",
            truth,
            corrupted,
            truth_adata.obs["cell_type"].astype(str).to_numpy(),
            split,
            truth_adata.var["feature_name"].astype(str).tolist(),
            {gene: (lambda labels, cell_types=cell_types: np.isin(labels, np.asarray(cell_types, dtype=str))) for gene, cell_types in PANELS["pancreas"]["source"].items()},
            {
                "safe_fusion": lambda pct, base=base: f"{base}/selector_mlp_biology_range_fullteachers/safe_fusion_calibrated_mlp_topk_{suffix(pct)}",
                "svd": lambda pct, base=base: f"{base}/matched_fraction/svd_topk_{suffix(pct)}",
                "weighted_knn": lambda pct, base=base: f"{base}/matched_fraction/weighted_knn_topk_{suffix(pct)}",
            },
            cell_ids,
        )
        summaries.append(result[0])
        markers.append(result[1])

    summary = pd.concat(summaries, ignore_index=True)
    per_marker = pd.concat(markers, ignore_index=True)
    summary.to_csv(output / "fill_set_summary.csv", index=False)
    per_marker.to_csv(output / "per_marker_offtarget.csv", index=False)

    per_marker["tissue"] = per_marker["dataset"].str.replace(r"_fold_\d", "", regex=True)
    marker_table = per_marker.groupby(["tissue", "marker"]).agg(
        nontarget_cells_detected=("nontarget_cells_detected", "mean"),
        masked_share_of_offtarget_zeros=("masked_share_of_offtarget_zeros", "mean"),
        n_offtarget_zeros=("n_offtarget_zeros", "sum"),
        safe_fusion=("safe_fusion_offtarget_fill_10pct", "mean"),
        svd=("svd_offtarget_fill_10pct", "mean"),
        weighted_knn=("weighted_knn_offtarget_fill_10pct", "mean"),
    ).reset_index()
    marker_table.to_csv(output / "per_marker_offtarget_by_tissue.csv", index=False)
    summary["tissue"] = summary["dataset"].str.replace(r"_fold_\d", "", regex=True)
    report = {
        "tissue_summary": summary.drop(columns="dataset").groupby(["tissue", "fill_pct"]).mean().reset_index().to_dict(orient="records"),
        "markers_fewer_offtarget_than_svd": {
            tissue: {"fewer": int((frame.safe_fusion < frame.svd).sum()), "total": int(len(frame))}
            for tissue, frame in marker_table.groupby("tissue")
        },
        "spearman_detected_outside_vs_offtarget_fill": {
            tissue: {
                method: float(frame["nontarget_cells_detected"].corr(frame[method], method="spearman"))
                for method in ("safe_fusion", "svd", "weighted_knn")
            }
            for tissue, frame in marker_table.groupby("tissue")
        },
    }
    (output / "tissue_report.json").write_text(json.dumps(report, indent=2))


def partial_spearman(first, second, control) -> float:
    ranks = [pd.Series(values).rank().to_numpy() for values in (first, second, control)]
    residuals = [values - np.polyval(np.polyfit(ranks[2], values, 1), ranks[2]) for values in ranks[:2]]
    return float(np.corrcoef(*residuals)[0, 1])


def papalexi_cd274(output: Path) -> None:
    truth_adata = ad.read_h5ad("external_data/prepared/papalexi_eccite_crossmodal.h5ad")
    corrupted_adata = ad.read_h5ad(f"{PAPALEXI}/corrupted.h5ad")
    truth = dense(truth_adata.layers["counts"] if "counts" in truth_adata.layers else truth_adata.X)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"])
    cells = pd.DataFrame({
        "cell_id": truth_adata.obs_names.astype(str),
        "source_cell_id": truth_adata.obs["source_cell_id"].astype(str).to_numpy(),
    })
    scores = pd.read_parquet(f"{PAPALEXI}/mlp_selector/selected_gene_scores.parquet")
    scores = scores.loc[(scores["split"].astype(str) == "test") & (scores["gene_id"].astype(str) == "CD274")]
    panel = pd.read_parquet("artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet")
    panel = panel.loc[panel["protein"].astype(str) == "PDL1", ["cell_id", "adt_clr"]].rename(columns={"cell_id": "source_cell_id"})
    frame = scores.merge(cells, on="cell_id").merge(panel, on="source_cell_id")
    gene = truth_adata.var_names.astype(str).tolist().index("CD274")
    index = frame["cell_index"].to_numpy(dtype=int)
    frame["truth"] = truth[index, gene]
    for name, directory in {"svd": "svd_impute", "weighted_knn": "graph_smooth", "fused_value": "safe_fusion", "scvi": "scvi"}.items():
        frame[name] = np.load(f"{PAPALEXI}/{directory}/mean.npy", mmap_mode="r")[index, gene]
    frame["log_library_size"] = np.log1p(corrupted.sum(axis=1))[index]
    frame = frame.loc[(frame["truth"] == 0) & (frame["masked_positive"].astype(int) == 0)]
    columns = ["selector_score", "svd", "weighted_knn", "fused_value", "scvi", "log_library_size", "adt_clr"]
    frame[columns].corr(method="spearman").to_csv(output / "cd274_ranking_spearman.csv")
    report = {
        "n_cells": int(len(frame)),
        "partial_spearman_selector_pdl1_given_svd": partial_spearman(frame.selector_score, frame.adt_clr, frame.svd),
        "partial_spearman_svd_pdl1_given_selector": partial_spearman(frame.svd, frame.adt_clr, frame.selector_score),
    }
    (output / "cd274_report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/paper_evidence/ranking_comparison")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tissues(output)
    papalexi_cd274(output)


if __name__ == "__main__":
    main()
