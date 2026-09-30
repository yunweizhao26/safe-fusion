#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = {
    "Safe Fusion": "#2a78d6",
    "Safe Fusion without MAGIC teacher": "#eb6834",
    "MAGIC": "#1baf7a",
    "scVI": "#eda100",
    "SAVER": "#e87ba4",
    "ALRA": "#008300",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

def series(root: Path) -> dict[str, pd.DataFrame]:
    totals = pd.read_csv(root / "results" / "safe_fusion_totals.csv")
    totals = totals[totals["completed"] == totals["steps"]].rename(
        columns={"wall_minutes_sum": "wall_minutes", "peak_memory_gb_max": "peak_memory_gb"})
    resources = pd.read_csv(root / "results" / "resources_fastest_run.csv")
    done = resources[(resources["state"] == "COMPLETED") & (resources["group"] == "Comparison")]
    result = {}
    for label in SERIES:
        if label in set(totals["variant"]):
            result[label] = totals[totals["variant"] == label].sort_values("cells")
        else:
            result[label] = done[done["label"] == label].sort_values("cells")
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/paper_evidence/review_round3/scale"))
    args = parser.parse_args()
    lines = series(args.root)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), constrained_layout=True)
    panels = (("wall_minutes", "Wall time (minutes)"), ("peak_memory_gb", "Peak memory (GB)"))
    sizes = sorted(set().union(*[frame["cells"] for frame in lines.values()]))
    for axis, (column, label) in zip(axes, panels):
        for name, frame in lines.items():
            if frame.empty:
                continue
            axis.plot(frame["cells"], frame[column], color=SERIES[name], linewidth=2, solid_capstyle="round",
                      marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5, label=name, zorder=3)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(sizes)
        axis.set_xticklabels([f"{size // 1000:,}k" for size in sizes])
        axis.minorticks_off()
        axis.set_xlabel("Cells")
        axis.set_ylabel(label)
        axis.grid(True, color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    output = args.root / "results" / "scaling_cost.png"
    figure.savefig(output, dpi=300)
    print(output)

if __name__ == "__main__":
    main()
