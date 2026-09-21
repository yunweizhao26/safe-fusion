#!/usr/bin/env python3
"""Plot the five downstream tasks across the complete Safe Fusion range."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path("artifacts/paper_evidence/downstream_complete/summary")

PANELS = [
    ("clustering", "annotation_ari", "Cell type clustering", "ARI change"),
    ("markers", "canonical_marker_pr_auc", "Marker recovery", "PR AUC change"),
    ("differential_expression", "disease_logfc_spearman", "Differential expression", "Spearman change"),
    ("trajectory", "dynamic_gene_spearman", "Trajectory", "Dynamic gene change"),
    ("grn", "edge_pr_auc_q10", "Interventional GRN", "Edge PR AUC change"),
]

COLORS = {
    "Pancreas": "#7B2CBF",
    "Colon": "#0072B2",
    "T1D": "#D55E00",
    "AAB": "#E69F00",
    "Zebrafish": "#009E73",
    "Norman CRISPRa": "#CC79A7",
    "Adamson CRISPRi": "#56B4E9",
    "Dixit knockout": "#E41A1C",
    "Papalexi ECCITE seq": "#8C564B",
}


def display_label(task: str, dataset: str, contrast: str | float) -> str:
    if task == "differential_expression":
        return "T1D" if str(contrast).startswith("T1D") else "AAB"
    return {
        "pancreas": "Pancreas",
        "colon": "Colon",
        "zebrafish": "Zebrafish",
        "norman_crispra": "Norman CRISPRa",
        "adamson_crispri": "Adamson CRISPRi",
        "dixit_ko": "Dixit knockout",
        "papalexi_eccite": "Papalexi ECCITE seq",
    }[dataset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path)
    args = parser.parse_args()
    comparisons = pd.read_parquet(ROOT / "all_paired_comparisons.parquet")
    comparisons = comparisons[comparisons["reference"] == "corrupted_raw"].copy()
    comparisons["budget"] = comparisons["method"].str.extract(r"safe_fusion_(\d+)pct")[0].astype(float)
    comparisons = comparisons[comparisons["budget"].notna()]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.6,
    })
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.8), dpi=200)
    axes = axes.ravel()
    legend_labels: list[str] = []
    for panel_index, (task, metric, title, ylabel) in enumerate(PANELS):
        axis = axes[panel_index]
        block = comparisons[(comparisons["task"] == task) & (comparisons["metric"] == metric)]
        group_columns = ["dataset"] + (["contrast"] if task == "differential_expression" else [])
        for keys, group in block.groupby(group_columns, dropna=False, observed=True):
            keys = keys if isinstance(keys, tuple) else (keys,)
            dataset = str(keys[0])
            contrast = keys[1] if len(keys) > 1 else np.nan
            label = display_label(task, dataset, contrast)
            group = group.sort_values("budget")
            if group["budget"].tolist() != list(range(1, 11)):
                raise ValueError(f"incomplete selected-fraction range for {task}/{label}")
            color = COLORS[label]
            axis.plot(group["budget"], group["difference"], color=color, linewidth=0.9)
            axis.fill_between(
                group["budget"].to_numpy(dtype=float),
                group["ci_low"].to_numpy(dtype=float),
                group["ci_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
            if label not in legend_labels:
                legend_labels.append(label)
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_title(title, fontweight="normal")
        axis.set_xlabel("Zeros selected (%)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([1, 2, 4, 6, 8, 10])
        axis.grid(True, color="#D0D0D0", alpha=0.25, linewidth=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[5].axis("off")
    handles = [Line2D([0], [0], color=COLORS[label], linewidth=0.9) for label in legend_labels]
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncol=5,
        frameon=False,
        handlelength=2.3,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.2, h_pad=1.4)
    output_png = ROOT / "complete_downstream_5panel.png"
    output_pdf = ROOT / "complete_downstream_5panel.pdf"
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    if args.paper_dir is not None:
        args.paper_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            args.paper_dir / "complete_downstream_5panel.png",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(output_png)


if __name__ == "__main__":
    main()
