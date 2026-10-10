#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

METHOD_STYLES = {
    "Safe Fusion": {"color": "#7B2CBF", "linewidth": 1.1, "linestyle": "-", "zorder": 10},
    "Safe Fusion MLP": {"color": "#7B2CBF", "linewidth": 1.1, "linestyle": "-", "zorder": 10},
    "scVI": {"color": "#0072B2", "linewidth": 0.9, "linestyle": "-", "zorder": 8},
    "MAGIC": {"color": "#D55E00", "linewidth": 0.9, "linestyle": "-", "zorder": 7},
    "SVD": {"color": "#009E73", "linewidth": 0.9, "linestyle": "-", "zorder": 6},
    "ALRA": {"color": "#E69F00", "linewidth": 0.9, "linestyle": "-", "zorder": 9},
    "SAVER": {"color": "#CC79A7", "linewidth": 0.9, "linestyle": "-", "zorder": 5},
    "scGPT": {"color": "#8C564B", "linewidth": 0.9, "linestyle": "-", "zorder": 3},
    "Weighted kNN": {"color": "#4D4D4D", "linewidth": 0.9, "linestyle": "--", "zorder": 2},
}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "artifacts"
        / "paper_evidence"
        / "selector_f1_fillrate_baselines_1000_points.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "paper_evidence"
        / "figures"
        / "f1_fillrate_3panel_oup.png",
    )
    parser.add_argument(
        "--safe-fusion-method",
        default=None,
        help="Method name plotted as the solid Safe Fusion curve (default: autodetect "
        "'Safe Fusion MLP' then 'Safe Fusion').",
    )
    parser.add_argument(
        "--second-curve-method",
        default=None,
        help="An additional method plotted as a dashed second curve in the Safe Fusion colour "
        "(for example the inductive selector, for reference next to a transductive main curve).",
    )
    parser.add_argument(
        "--second-curve-label",
        default=None,
        help="Legend label of --second-curve-method (default: the method name).",
    )
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    datasets = ("Pancreas", "Colon", "CRISPRa")
    if args.safe_fusion_method is not None:
        safe_fusion_method = args.safe_fusion_method
    else:
        safe_fusion_method = (
            "Safe Fusion MLP"
            if "Safe Fusion MLP" in set(table["method"])
            else "Safe Fusion"
        )

    methods = (
        safe_fusion_method,
        "scVI",
        "MAGIC",
        "SVD",
        "ALRA",
        "SAVER",
        "scGPT",
        "Weighted kNN",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)

    for axis, dataset in zip(axes, datasets):
        subset = table.loc[table["dataset"] == dataset]
        for method in methods:
            style = METHOD_STYLES.get(method, METHOD_STYLES["Safe Fusion"])
            curve = subset.loc[subset["method"] == method]
            if curve.empty:
                raise ValueError(f"Missing {dataset} curve for {method}")
            axis.plot(
                100.0 * curve["realized_fill_fraction"],
                100.0 * curve["masked_f1"],
                label="Safe Fusion" if method in ("Safe Fusion MLP", safe_fusion_method) else method,
                **style,
            )
        if args.second_curve_method is not None:
            style = dict(METHOD_STYLES["Safe Fusion"])
            style["linestyle"] = "--"
            style["zorder"] = style.get("zorder", 10) - 1
            curve = subset.loc[subset["method"] == args.second_curve_method]
            if curve.empty:
                raise ValueError(f"Missing {dataset} curve for {args.second_curve_method}")
            axis.plot(
                100.0 * curve["realized_fill_fraction"],
                100.0 * curve["masked_f1"],
                label=args.second_curve_label or args.second_curve_method,
                **style,
            )
        axis.set_title(dataset)
        axis.set_xlabel("fill fraction (% of candidate zeros)")
        axis.set_xscale("log")
        axis.set_xlim(0.1, 100.0)
        axis.set_xticks([0.1, 1, 10, 100])
        axis.set_xticklabels(["0.1", "1", "10", "100"])
        axis.grid(alpha=0.22)

    axes[0].set_ylabel("masked F1 (%)")
    axes[0].set_ylim(bottom=0.0)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=8,
        loc="lower center",
        ncol=len(methods) + (1 if args.second_curve_method is not None else 0),
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=240, bbox_inches="tight")

if __name__ == "__main__":
    main()
