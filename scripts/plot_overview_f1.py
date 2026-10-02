from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STYLES = {
    "Safe Fusion (transductive)": {"label": "Safe Fusion", "color": "#7B2CBF", "linewidth": 1.3, "linestyle": "-", "zorder": 10},
    "Safe Fusion (inductive)": {"label": "Safe Fusion, inductive", "color": "#7B2CBF", "linewidth": 1.0, "linestyle": "--", "zorder": 9},
    "scVI": {"label": "scVI", "color": "#0072B2", "linewidth": 0.9, "linestyle": "-", "zorder": 8},
    "MAGIC": {"label": "MAGIC", "color": "#D55E00", "linewidth": 0.9, "linestyle": "-", "zorder": 7},
    "SVD": {"label": "SVD", "color": "#009E73", "linewidth": 0.9, "linestyle": "-", "zorder": 6},
    "ALRA": {"label": "ALRA", "color": "#E69F00", "linewidth": 0.9, "linestyle": "-", "zorder": 5},
    "SAVER": {"label": "SAVER", "color": "#CC79A7", "linewidth": 0.9, "linestyle": "-", "zorder": 4},
    "scGPT": {"label": "scGPT", "color": "#8C564B", "linewidth": 0.9, "linestyle": "-", "zorder": 3},
    "Weighted kNN": {"label": "Weighted kNN", "color": "#4D4D4D", "linewidth": 0.9, "linestyle": ":", "zorder": 2},
}
DATASETS = ("Pancreas", "Colon", "CRISPRa")
BOX_FACE = "#F3EEF9"
BOX_EDGE = "#7B2CBF"
TEXT = "#222222"

def box(axis, x, y, width, height, text, fontsize=7.6):
    patch = FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor=BOX_FACE, edgecolor=BOX_EDGE, linewidth=0.9,
    )
    axis.add_patch(patch)
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=TEXT, linespacing=1.25)

def arrow(axis, start, end, dashed=False):
    axis.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
            color="#555555", linestyle="--" if dashed else "-",
        )
    )

def draw_overview(axis):
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    box(axis, 0.10, 0.84, 0.80, 0.13, "Count matrix\nzeros are the candidates")
    box(axis, 0.10, 0.59, 0.80, 0.16, "Five teachers propose a count\nmedian, SVD, kNN, MAGIC, scVI")
    box(axis, 0.02, 0.29, 0.45, 0.20, "Fused value\nboosted regression\nof the proposals")
    box(axis, 0.53, 0.29, 0.45, 0.20, "Selector\nnetwork trained on\nhidden counts")
    box(axis, 0.10, 0.03, 0.80, 0.16, "Fill the top-ranked fraction of zeros\nnonzero counts stay unchanged")
    arrow(axis, (0.50, 0.84), (0.50, 0.75))
    arrow(axis, (0.32, 0.59), (0.25, 0.49))
    arrow(axis, (0.68, 0.59), (0.75, 0.49))
    arrow(axis, (0.25, 0.29), (0.32, 0.19))
    arrow(axis, (0.75, 0.29), (0.68, 0.19))
    axis.text(0.245, 0.535, "value", fontsize=6.6, color="#555555", ha="right", va="center")
    axis.text(0.755, 0.535, "score", fontsize=6.6, color="#555555", ha="left", va="center")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves",
        type=Path,
        default=Path("artifacts/paper_evidence/review_round4/transductive_main/selector_f1_fillrate_transductive_1000_points.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/paper_evidence/figures/overview_f1_fillrate.png"))
    args = parser.parse_args()

    curves = pd.read_csv(args.curves)
    figure, axes = plt.subplots(1, 4, figsize=(12, 2.85), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    draw_overview(axes[0])
    for axis, dataset in zip(axes[1:], DATASETS):
        subset = curves[curves["dataset"] == dataset]
        for method, style in STYLES.items():
            rows = subset[subset["method"] == method].sort_values("requested_fill_fraction")
            if rows.empty:
                continue
            axis.plot(
                100 * rows["requested_fill_fraction"], 100 * rows["masked_f1"], label=style["label"],
                color=style["color"], linewidth=style["linewidth"], linestyle=style["linestyle"], zorder=style["zorder"],
            )
        axis.set_xscale("log")
        axis.set_xlim(0.1, 100.0)
        axis.set_ylim(bottom=0.0)
        axis.set_xticks([0.1, 1, 10, 100])
        axis.set_xticklabels(["0.1", "1", "10", "100"])
        axis.set_title(dataset, fontsize=10)
        axis.set_xlabel("fill fraction (% of candidate zeros)", fontsize=8.5)
        axis.grid(alpha=0.25, linewidth=0.5)
        axis.tick_params(labelsize=8)
    axes[1].set_ylabel("masked F1 (%)", fontsize=8.5)
    for axis in axes[2:]:
        axis.sharey(axes[1])
        axis.tick_params(labelleft=False)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=7.6, bbox_to_anchor=(0.62, 0.0))
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=240)

if __name__ == "__main__":
    main()
