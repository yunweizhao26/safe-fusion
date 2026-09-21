#!/usr/bin/env python3
"""Evaluate prespecified epithelial and inflammation markers on locked donors."""

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
from safefusion_benchmark.metrics import average_precision_tie_aware, log1p_mae, spearman  # noqa: E402


CELL_MARKERS = {
    "BEST4": "Enterocytes BEST4", "OTOP2": "Enterocytes BEST4",
    "CA7": "Enterocytes BEST4", "GUCA2A": "Enterocytes BEST4", "GUCA2B": "Enterocytes BEST4",
    "CA1": "Enterocytes CA1", "CA2": "Enterocytes CA1",
    "TMIGD1": "Enterocytes TMIGD1", "MEP1A": "Enterocytes TMIGD1",
    "MUC2": "Goblet", "TFF3": "Goblet", "AGR2": "Goblet", "SPDEF": "Goblet",
    "TFF1": "Goblet cells MUC2 TFF1", "SPINK4": "Goblet cells SPINK4",
    "POU2F3": "Tuft", "DCLK1": "Tuft", "CHGA": "Enteroendocrine",
    "GCG": "Enteroendocrine", "GIP": "Enteroendocrine", "CCK": "Enteroendocrine",
    "LYZ": "Paneth", "DEFA5": "Paneth", "MKI67": "Cycling", "PCNA": "Cycling",
    "OLFM4": "Stem", "LGR5": "Stem", "ASCL2": "Stem", "SOX9": "Stem",
}

INFLAMMATION_MARKERS = {
    "REG1A", "REG3A", "DUOX2", "NOS2", "CXCL1", "CXCL2", "CXCL3", "CXCL8",
    "CCL20", "IL32", "HLA-DRA", "HLA-DPA1", "HLA-DPB1", "HLA-A", "HLA-B",
    "STAT1", "IRF1", "IFITM1", "IFITM3",
}

PANCREAS_CELL_MARKERS = {
    **{gene: ("beta_major", "beta_minor") for gene in ("INS", "IAPP", "PCSK1", "PCSK2", "MAFA", "NKX6-1", "PDX1")},
    **{gene: ("alpha",) for gene in ("GCG", "TTR")},
    "SST": ("delta",), "PPY": ("pp",), "GHRL": ("epsilon",),
    **{gene: ("acinar", "acinar_minor_mhcclassII", "duct_acinar_related") for gene in ("PRSS1", "PRSS2", "REG1A", "REG1B", "CPA1", "CTRB2")},
    **{gene: ("duct_major", "duct_acinar_related") for gene in ("KRT8", "KRT18", "KRT19", "MUC1", "KRT17", "KRT7")},
    **{gene: ("stellates", "immune_stellates") for gene in ("COL1A1", "COL1A2", "COL3A1", "SPARC", "DCN")},
    **{gene: ("endothelial",) for gene in ("KDR", "EMCN", "PLVAP", "VWF")},
    **{gene: ("immune_stellates",) for gene in ("PTPRC", "CD74", "HLA-DRA", "HLA-DPA1", "HLA-DPB1")},
}

PANCREAS_DISEASE_MARKERS = {"CXCL10", "STAT1", "B2M", "IFITM1", "IFITM3"}


def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def parse_method(value: str) -> tuple[str, Path]:
    name, path = value.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--method", action="append", required=True, help="name=contract_directory")
    parser.add_argument("--panel", choices=["colon", "pancreas"], default="colon")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    truth_adata = ad.read_h5ad(args.truth)
    corrupted_adata = ad.read_h5ad(args.corrupted)
    truth = dense(truth_adata.layers["counts"]).astype(np.float32)
    corrupted = dense(corrupted_adata.layers["corrupted_counts"]).astype(np.float32)
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[truth_adata.obs_names.astype(str), "split"].to_numpy()
    test = split == "test"
    feature_name = truth_adata.var["feature_name"].astype(str).to_numpy()
    gene_lookup = {gene: index for index, gene in enumerate(feature_name)}
    cell_markers = CELL_MARKERS if args.panel == "colon" else PANCREAS_CELL_MARKERS
    condition_markers = INFLAMMATION_MARKERS if args.panel == "colon" else PANCREAS_DISEASE_MARKERS
    positive_conditions = {"inflamed"} if args.panel == "colon" else {"T1D"}
    selected_symbols = sorted((set(cell_markers) | condition_markers) & set(gene_lookup))
    selected_indices = np.asarray([gene_lookup[gene] for gene in selected_symbols], dtype=int)

    coordinates = pd.read_parquet(args.coordinates, filters=[("split", "==", "test")])
    coordinate_symbols = feature_name[coordinates["gene_index"].to_numpy(dtype=int)]
    coordinates = coordinates.assign(gene_symbol=coordinate_symbols)
    marker_coordinates = coordinates[coordinates["gene_symbol"].isin(selected_symbols)].copy()
    row_index = marker_coordinates["cell_index"].to_numpy(dtype=int)
    col_index = marker_coordinates["gene_index"].to_numpy(dtype=int)
    marker_truth = marker_coordinates["original_value"].to_numpy(dtype=float)

    outputs = {"corrupted_raw": corrupted}
    for specification in args.method:
        name, path = parse_method(specification)
        outputs[name] = np.load(path / "mean.npy", allow_pickle=False)

    records: list[dict] = []
    for name, matrix in outputs.items():
        marker_prediction = matrix[row_index, col_index]
        records.append({
            "method": name, "task": "masked_curated_markers", "scope": "all_markers",
            "marker": "all", "metric": "log1p_mae", "value": log1p_mae(marker_truth, marker_prediction),
            "n": int(len(marker_truth)),
        })
        for marker, block in marker_coordinates.groupby("gene_symbol", observed=True, sort=True):
            positions = block.index.to_numpy()
            local = marker_coordinates.index.get_indexer(positions)
            records.append({
                "method": name, "task": "masked_curated_markers", "scope": "marker",
                "marker": marker, "metric": "log1p_mae",
                "value": log1p_mae(marker_truth[local], marker_prediction[local]), "n": int(len(local)),
            })

        specificity_truth: list[float] = []
        specificity_predicted: list[float] = []
        auc_values: list[float] = []
        ectopic: list[float] = []
        test_labels = truth_adata.obs.loc[test, "cell_type"].astype(str).to_numpy()
        test_condition = truth_adata.obs.loc[test, "condition"].astype(str).to_numpy()
        for marker in selected_symbols:
            gene = gene_lookup[marker]
            if marker in cell_markers:
                targets = cell_markers[marker]
                if isinstance(targets, str):
                    labels = np.char.find(test_labels.astype(str), targets) >= 0
                else:
                    labels = np.isin(test_labels, np.asarray(targets, dtype=str))
            else:
                labels = np.isin(test_condition, np.asarray(sorted(positive_conditions), dtype=str))
            if not labels.any() or labels.all():
                continue
            true_values = truth[test, gene]
            predicted_values = matrix[test, gene]
            true_effect = float(np.log1p(true_values[labels].mean()) - np.log1p(true_values[~labels].mean()))
            predicted_effect = float(np.log1p(predicted_values[labels].mean()) - np.log1p(predicted_values[~labels].mean()))
            specificity_truth.append(true_effect)
            specificity_predicted.append(predicted_effect)
            auc_values.append(average_precision_tie_aware(labels, predicted_values))
            off_target_original_zero = (~labels) & (true_values == 0)
            if off_target_original_zero.any():
                ectopic.append(float(np.mean(predicted_values[off_target_original_zero] > 0.5)))
        records.extend([
            {"method": name, "task": "marker_specificity", "scope": "all_markers", "marker": "all", "metric": "effect_spearman", "value": spearman(np.asarray(specificity_truth), np.asarray(specificity_predicted)), "n": len(specificity_truth)},
            {"method": name, "task": "marker_specificity", "scope": "all_markers", "marker": "all", "metric": "mean_pr_auc", "value": float(np.mean(auc_values)), "n": len(auc_values)},
            {"method": name, "task": "ectopic_marker_induction", "scope": "all_markers", "marker": "all", "metric": "off_target_zero_fill_rate_gt_0.5", "value": float(np.mean(ectopic)), "n": len(ectopic)},
        ])

    result = pd.DataFrame(records)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False)
    print(json.dumps({"output": str(destination), "rows": len(result), "markers": selected_symbols, "masked_marker_coordinates": len(marker_coordinates)}))


if __name__ == "__main__":
    main()
