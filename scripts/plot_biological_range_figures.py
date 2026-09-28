#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SAFE = "#7B2CBF"
KNN = "#4D4D4D"
SVD = "#009E73"
SCVI = "#0072B2"
FUSED = "#D55E00"
LIBRARY = "#8C564B"
RAW = "#56B4E9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pancreas-marker-summary",
        type=Path,
        default=Path("artifacts/paper_evidence/downstream_complete/pancreas_biology/bootstrap_summary.parquet"),
    )
    parser.add_argument(
        "--colon-marker-summary",
        type=Path,
        default=Path("artifacts/paper_evidence/downstream_complete/markers/colon/bootstrap_summary.parquet"),
    )
    parser.add_argument(
        "--crossmodal-root",
        type=Path,
        default=Path("artifacts/paper_evidence/papalexi_crossmodal/benchmark/evaluation_cd274"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/paper_evidence/figures"),
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        help="Optional directory for an additional PNG copy.",
    )
    parser.add_argument(
        "--pdl1-only",
        action="store_true",
        help="Regenerate only the PD-L1 figure used by the manuscript.",
    )
    return parser.parse_args()


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def polish(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22)


def save_figure(fig: plt.Figure, stem: str, output_dir: Path, paper_dir: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    directories = [output_dir]
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        directories.append(paper_dir)
    for directory in directories:
        fig.savefig(directory / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)


def marker_rows(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    if "contrast" in data.columns:
        data = data.loc[data["contrast"].astype(str) == "all_conditions"]
    data = data.loc[
        (data["scope"].astype(str) == "donor")
        & data["metric"].isin(["canonical_marker_pr_auc", "ectopic_marker_fill_rate"])
    ].copy()
    wide = data.pivot(index="method", columns="metric", values=["estimate", "ci_low", "ci_high"])
    wide.columns = [f"{stat}_{metric}" for stat, metric in wide.columns]
    wide = wide.reset_index()
    wide["method"] = wide["method"].replace({"weighted_knn": "graph_smooth"})
    allowed = ["corrupted_raw", "graph_smooth"]
    allowed.extend(f"safe_fusion_{value}pct" for value in range(1, 11))
    allowed.extend(f"mlp_{value}pct" for value in range(1, 11))
    wide = wide.loc[wide["method"].isin(allowed)].copy()
    wide["selected_fraction"] = wide["method"].map(
        lambda value: int(re.search(r"(?:safe_fusion|mlp)_(\d+)pct", value).group(1))
        if value.startswith(("safe_fusion_", "mlp_"))
        else np.nan
    )
    return wide


def plot_marker_tradeoff(
    pancreas_summary: Path,
    colon_summary: Path,
    output_dir: Path,
    paper_dir: Path | None,
) -> dict:
    datasets = [
        ("Pancreas", pancreas_summary),
        ("Colon", colon_summary),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 4.8), sharex="col", sharey="row")
    summary: dict[str, dict] = {}
    for column, (dataset, path) in enumerate(datasets):
        data = marker_rows(path)
        safe = data.loc[
            data["method"].str.startswith(("safe_fusion_", "mlp_"))
        ].sort_values("selected_fraction")
        raw = data.loc[data["method"] == "corrupted_raw"].iloc[0]
        knn = data.loc[data["method"] == "graph_smooth"].iloc[0]
        selected = safe["selected_fraction"].to_numpy(dtype=float)
        marker = 100 * safe["estimate_canonical_marker_pr_auc"].to_numpy()
        off_target = 100 * safe["estimate_ectopic_marker_fill_rate"].to_numpy()
        assert np.array_equal(selected, np.arange(1.0, 11.0))

        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(selected, marker, color=SAFE, linewidth=0.9, label="Safe Fusion")
        top.plot(selected, np.repeat(100 * raw["estimate_canonical_marker_pr_auc"], len(selected)), color=RAW, linewidth=0.9, label="Masked input")
        top.plot(selected, np.repeat(100 * knn["estimate_canonical_marker_pr_auc"], len(selected)), color=KNN, linewidth=0.9, label="Weighted kNN")
        bottom.plot(selected, off_target, color=SAFE, linewidth=0.9, label="Safe Fusion")
        bottom.plot(selected, np.repeat(100 * raw["estimate_ectopic_marker_fill_rate"], len(selected)), color=RAW, linewidth=0.9, label="Masked input")
        bottom.plot(selected, np.repeat(100 * knn["estimate_ectopic_marker_fill_rate"], len(selected)), color=KNN, linewidth=0.9, label="Weighted kNN")
        top.set_title(dataset)
        bottom.set_xlabel("zeros selected (%)")
        top.set_xlim(1, 10)
        top.set_xticks(np.arange(1, 11))
        for axis in (top, bottom):
            polish(axis)
        summary[dataset.lower()] = {
            "safe_fusion_marker_pr_auc_percent": [float(marker.min()), float(marker.max())],
            "safe_fusion_off_target_fill_percent": [float(off_target.min()), float(off_target.max())],
            "masked_input_marker_pr_auc_percent": float(100 * raw["estimate_canonical_marker_pr_auc"]),
            "weighted_knn_marker_pr_auc_percent": float(100 * knn["estimate_canonical_marker_pr_auc"]),
            "weighted_knn_off_target_fill_percent": float(100 * knn["estimate_ectopic_marker_fill_rate"]),
        }
    axes[0, 0].set_ylabel("marker PR AUC (%)")
    axes[1, 0].set_ylabel("off target zeros filled (%)")
    axes[0, 0].set_ylim(30, 72)
    axes[1, 0].set_ylim(-0.4, 19)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 1.0))
    save_figure(fig, "marker_recovery_tradeoff", output_dir, paper_dir)
    plt.close(fig)
    return summary


def plot_pdl1_ranges(crossmodal_root: Path, output_dir: Path, paper_dir: Path | None) -> dict:
    thresholds = pd.read_csv(crossmodal_root / "protein_threshold_range_metrics.csv")
    fills = pd.read_csv(crossmodal_root / "fill_range_protein_enrichment.csv")
    method_order = ["mlp_safe_fusion", "svd", "weighted_knn", "scvi", "fused_component", "library_size"]
    labels = {
        "mlp_safe_fusion": "Safe Fusion",
        "svd": "SVD",
        "weighted_knn": "Weighted kNN",
        "scvi": "scVI",
        "fused_component": "Fused value",
        "library_size": "Library size",
    }
    colors = {
        "mlp_safe_fusion": SAFE,
        "svd": SVD,
        "weighted_knn": KNN,
        "scvi": SCVI,
        "fused_component": FUSED,
        "library_size": LIBRARY,
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.4))
    for method in method_order:
        current = thresholds.loc[thresholds["method"] == method].sort_values("protein_threshold_quantile")
        axes[0].plot(
            100 * current["protein_threshold_quantile"],
            100 * current["roc_auc"],
            color=colors[method],
            linewidth=0.9,
            label=labels[method],
        )
        current = fills.loc[fills["method"] == method].sort_values("selected_fraction")
        axes[1].plot(
            100 * current["selected_fraction"],
            current["protein_enrichment"],
            color=colors[method],
            linewidth=0.9,
            label=labels[method],
        )
    axes[0].axhline(50, color="#A8ADB4", linewidth=0.9, zorder=0)
    axes[0].set_xlabel("PD-L1 threshold percentile")
    axes[0].set_ylabel("AUROC (%)  ↑")
    axes[0].set_xlim(30, 70)
    axes[0].set_ylim(48, 81)
    axes[1].axhline(0, color="#A8ADB4", linewidth=0.9, zorder=0)
    axes[1].set_xlabel("CD274 RNA zeros filled (%)")
    axes[1].set_ylabel("Mean PD-L1 enrichment  ↑")
    axes[1].set_xlim(1, 20)
    axes[1].set_xticks([1, 5, 10, 15, 20])
    for ax in axes:
        polish(ax)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    save_figure(fig, "pdl1_range_validation", output_dir, paper_dir)
    plt.close(fig)

    threshold_leaders = (
        thresholds.loc[thresholds.groupby("protein_threshold_quantile")["roc_auc"].idxmax(), "method"]
        .value_counts()
        .to_dict()
    )
    fill_leaders = (
        fills.loc[fills.groupby("selected_fraction")["protein_enrichment"].idxmax(), "method"]
        .value_counts()
        .to_dict()
    )
    safe_threshold = thresholds.loc[thresholds["method"] == "mlp_safe_fusion", "roc_auc"]
    return {
        "safe_fusion_mean_roc_auc_percent": float(100 * safe_threshold.mean()),
        "safe_fusion_roc_auc_range_percent": [
            float(100 * safe_threshold.min()),
            float(100 * safe_threshold.max()),
        ],
        "threshold_leaders": {key: int(value) for key, value in threshold_leaders.items()},
        "fill_fraction_leaders": {key: int(value) for key, value in fill_leaders.items()},
    }


def main() -> None:
    args = parse_args()
    set_style()
    summary = {}
    if not args.pdl1_only:
        summary["marker_tradeoff"] = plot_marker_tradeoff(
            args.pancreas_marker_summary,
            args.colon_marker_summary,
            args.output_dir,
            args.paper_dir,
        )
    summary["pdl1_ranges"] = plot_pdl1_ranges(
        args.crossmodal_root,
        args.output_dir,
        args.paper_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "biological_range_figures.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
