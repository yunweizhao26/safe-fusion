from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

STYLES = {
    "Safe Fusion (transductive)": {"label": "Safe Fusion", "color": "#7B2CBF", "linewidth": 2.0, "linestyle": "-", "zorder": 10},
    "Safe Fusion (inductive)": {"label": "Safe Fusion, inductive", "color": "#7B2CBF", "linewidth": 1.3, "linestyle": "--", "zorder": 9},
    "scVI": {"label": "scVI", "color": "#0072B2", "linewidth": 1.1, "linestyle": "-", "zorder": 8},
    "MAGIC": {"label": "MAGIC", "color": "#D55E00", "linewidth": 1.1, "linestyle": "-", "zorder": 7},
    "SVD": {"label": "SVD", "color": "#009E73", "linewidth": 1.1, "linestyle": "-", "zorder": 6},
    "ALRA": {"label": "ALRA", "color": "#E69F00", "linewidth": 1.1, "linestyle": "-", "zorder": 5},
    "SAVER": {"label": "SAVER", "color": "#CC79A7", "linewidth": 1.1, "linestyle": "-", "zorder": 4},
    "scGPT": {"label": "scGPT", "color": "#8C564B", "linewidth": 1.1, "linestyle": "-", "zorder": 3},
    "Weighted kNN": {"label": "Weighted kNN", "color": "#4D4D4D", "linewidth": 1.1, "linestyle": ":", "zorder": 2},
}
DATASETS = ("Pancreas", "Colon", "CRISPRa")
INK = "#2B2B2B"
MUTED = "#6B6B6B"


def style_axis(axis):
    axis.set_xscale("log")
    axis.set_xlim(0.1, 100.0)
    axis.set_xticks([0.1, 1, 10, 100])
    axis.set_xticklabels(["0.1", "1", "10", "100"])
    axis.tick_params(labelsize=10.5, colors=MUTED)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#BDBDBD")
    axis.grid(axis="y", color="#EDEDED", linewidth=0.6, zorder=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curves",
        type=Path,
        default=Path("artifacts/paper_evidence/review_round4/round2_extras/part_b/selector_f1_fillrate_allcell_1000_points.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/paper_evidence/figures/overview_f1_fillrate.pdf"))
    args = parser.parse_args()

    curves = pd.read_csv(args.curves)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for axis, dataset in zip(axes, DATASETS):
        table = curves[curves["dataset"] == dataset].pivot(index="requested_fill_fraction", columns="method", values="masked_f1")
        fill = 100 * table.index.to_numpy()
        for method, style in STYLES.items():
            if method not in table:
                continue
            axis.plot(fill, 100 * table[method], label=style["label"], color=style["color"], linewidth=style["linewidth"],
                      linestyle=style["linestyle"], zorder=style["zorder"])
        axis.set_ylim(bottom=0.0)
        axis.set_title(dataset, fontsize=12.5, color=INK)
        axis.set_xlabel("Fill fraction (% of candidate zeros)", fontsize=11, color=INK)
        style_axis(axis)
    axes[0].set_ylabel("Masked F1 (%)", fontsize=11, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=10.5,
                  bbox_to_anchor=(0.5, 0.0), handlelength=2.2, columnspacing=1.4, labelcolor=INK)
    figure.tight_layout(rect=(0, 0.09, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, facecolor="white")
    figure.savefig(args.output.with_suffix(".png"), dpi=240, facecolor="white")


if __name__ == "__main__":
    main()
