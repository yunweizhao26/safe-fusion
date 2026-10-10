import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

METHODS = [
    ("detection", "Safe Fusion", "#7B2CBF", 2.2, 10, "o"),
    ("magic", "MAGIC", "#D55E00", 1.1, 6, "^"),
    ("scvi", "scVI", "#0072B2", 1.1, 7, "D"),
    ("alra_all_cells", "ALRA", "#E69F00", 1.1, 5, "X"),
    ("saver", "SAVER", "#CC79A7", 1.1, 4, "v"),
    ("dca", "DCA", "#009E73", 1.1, 4, "s"),
    ("scimpute", "scImpute", "#56B4E9", 1.1, 3, "P"),
    ("screcover", "scRecover", "#8C564B", 1.1, 3, "h"),
    ("enimpute", "EnImpute", "#4D4D4D", 1.1, 4, "p"),
]
IMPROVE = [
    ("colon", "Marker AUPRC", "Marker recovery, colon"),
    ("pancreas", "Marker AUPRC", "Marker recovery, pancreas"),
    ("zebrafish", "Dynamics", "Developmental dynamics, zebrafish"),
    ("norman_crispra", "Edge AUPRC", "Response edges, Norman"),
]
WORSEN = [
    ("colon", "Reference mapping F1", "Reference mapping, colon"),
    ("pancreas", "Reference mapping F1", "Reference mapping, pancreas"),
    ("zebrafish", "Stage error", "Stage prediction, zebrafish"),
]
TICKS = [0.25, 1, 3, 10, 30, 100]
FLOOR = -150
INK = "#2B2B2B"
MUTED = "#6B6B6B"


def panel(ax, data, dataset, endpoint, title):
    sub = data[(data.dataset == dataset) & (data.endpoint == endpoint)]
    ax.axhline(0, color="#8C8C8C", linewidth=1.0, zorder=1)
    ax.axvline(1, color="#9A9A9A", linewidth=0.9, linestyle=":", zorder=1)
    for key, label, color, width, z, marker in METHODS:
        rows = sub[sub.method == key].sort_values("fill_pct")
        if rows.empty:
            continue
        if key == "detection":
            ax.fill_between(rows.fill_pct, rows.gap_ci_low_pct, rows.gap_ci_high_pct, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(rows.fill_pct, rows.gap_closed_pct, color=color, linewidth=width, zorder=z, marker=marker,
                markersize=3.2, markeredgewidth=0.7, markeredgecolor=color, markerfacecolor=color)
    shown = sub[sub.method.isin([m[0] for m in METHODS])]
    band = sub[sub.method == "detection"]
    low = min(shown.gap_closed_pct.min(), band.gap_ci_low_pct.min())
    high = max(shown.gap_closed_pct.max(), band.gap_ci_high_pct.max())
    bottom = max(low, FLOOR)
    ax.set_ylim(bottom - 0.06 * (high - bottom), high + 0.06 * (high - bottom))
    ax.set_xscale("log")
    ax.set_xticks(TICKS)
    ax.set_xticklabels([f"{t:g}" for t in TICKS])
    ax.minorticks_off()
    ax.set_xlim(0.22, 115)
    ax.set_title(title, fontsize=11.5, color=INK)
    ax.tick_params(labelsize=10.5, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#BDBDBD")
    ax.grid(axis="y", color="#EDEDED", linewidth=0.6, zorder=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, nargs="+", default=[
        Path("artifacts/paper_evidence/review_round4/published_downstream/fill_grid_published_long.csv"),
        Path("artifacts/paper_evidence/review_round4/published_downstream_allcells/fill_grid_allcells_long.csv"),
    ])
    parser.add_argument("--output", type=Path, default=Path("artifacts/paper_evidence/figures/fill_grid_main.png"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.data], ignore_index=True)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6))
    for ax, (dataset, endpoint, title) in zip(axes[0], IMPROVE):
        panel(ax, data, dataset, endpoint, title)
    for ax, (dataset, endpoint, title) in zip(axes[1, :3], WORSEN):
        panel(ax, data, dataset, endpoint, title)
    for ax in [*axes[1, :3], axes[0, 3]]:
        ax.set_xlabel("Fill fraction (%)", fontsize=11, color=INK)
    for row in range(2):
        axes[row, 0].set_ylabel("Gap to deeper data closed (%)", fontsize=11, color=INK)
    legend_ax = axes[1, 3]
    legend_ax.axis("off")
    handles = [Line2D([0], [0], color=c, linewidth=w, marker=m, markersize=5.5, markeredgecolor=c, markerfacecolor=c)
               for _, _, c, w, _, m in METHODS]
    handles.append(Line2D([0], [0], color="#9A9A9A", linestyle=":", linewidth=0.9))
    labels = [label for _, label, *_ in METHODS] + ["1% default fill"]
    legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=11, handlelength=2.4, labelcolor=INK)
    fig.text(0.012, 0.965, "Endpoints that improve with moderate filling", fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.487, "Endpoints that worsen with more filling", fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), facecolor="white")


if __name__ == "__main__":
    main()
