#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

NAME = re.compile(r"^(?P<source>.+?)_(?P<pct>\d+)pct(?:__(?P<part>masked_only|zeros_only))?$")
KEY_METRICS = {
    "canonical_marker_pr_auc",
    "ectopic_marker_fill_rate",
    "cell_identity_macro_f1",
    "annotation_ari",
    "neighbor_purity",
    "disease_logfc_spearman",
    "disease_logfc_rmse",
    "stage_rank_mae",
    "dynamic_gene_spearman",
    "stage_pseudobulk_spearman",
    "edge_pr_auc_q05",
    "edge_pr_auc_q10",
    "effect_spearman_test_truth",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True, help="CSV path for the tidy table")
    args = parser.parse_args()

    root = Path(args.root)
    frames = []
    for path in sorted(root.rglob("paired_comparisons.parquet")):
        frame = pd.read_parquet(path)
        frame = frame[frame["reference"] == "corrupted_raw"].copy()
        if frame.empty:
            continue
        frame["evaluation"] = str(path.parent.relative_to(root))
        frames.append(frame)
    if not frames:
        raise ValueError(f"no paired comparisons under {root}")
    table = pd.concat(frames, ignore_index=True)
    parsed = table["method"].str.extract(NAME)
    table["source"] = parsed["source"].fillna(table["method"])
    table["fill_pct"] = pd.to_numeric(parsed["pct"], errors="coerce")
    table["part"] = parsed["pct"].where(parsed["pct"].isna(), parsed["part"].fillna("full"))
    table["part"] = table["part"].fillna("dense")
    columns = [
        "evaluation", "scope", "contrast", "source", "fill_pct", "part", "method", "metric",
        "difference", "ci_low", "ci_high", "n_units",
    ]
    table = table[[column for column in columns if column in table.columns]]
    table = table.sort_values([c for c in ("evaluation", "metric", "contrast", "source", "fill_pct", "part") if c in table.columns])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    key = table[table["metric"].isin(KEY_METRICS)]
    key.to_csv(output.with_name(output.stem + "_key_metrics.csv"), index=False)
    print(f"{len(table)} rows, {len(key)} key-metric rows -> {output}")


if __name__ == "__main__":
    main()
