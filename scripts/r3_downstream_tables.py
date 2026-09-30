#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"
PUBLISHED_DEPLOYMENT = {
    "grn/norman_crispra": EVIDENCE / "review_round2/norman_rebuilt/deployment/evaluation/grn/norman_crispra",
}
PUBLISHED_MASKED = {
    "grn/norman_crispra": EVIDENCE / "review_round2/norman_rebuilt/downstream_masked/grn/norman_crispra",
}
ENDPOINTS = (
    ("Colon marker AUPRC", "markers/colon", "canonical_marker_pr_auc", None, 100.0),
    ("Colon off-target fill", "markers/colon", "ectopic_marker_fill_rate", None, 100.0),
    ("Colon reference mapping F1", "markers/colon", "cell_identity_macro_f1", None, 100.0),
    ("Colon adjusted Rand index", "clustering/colon", "annotation_ari", None, 1.0),
    ("Pancreas marker AUPRC", "pancreas_biology", "canonical_marker_pr_auc", None, 100.0),
    ("Pancreas off-target fill", "pancreas_biology", "ectopic_marker_fill_rate", None, 100.0),
    ("Pancreas reference mapping F1", "pancreas_biology", "cell_identity_macro_f1", None, 100.0),
    ("Pancreas adjusted Rand index", "clustering/pancreas/combined", "annotation_ari", None, 1.0),
    ("Pancreas disease effects, autoantibody", "pancreas_biology", "disease_logfc_spearman", "AAB_vs_Control", 1.0),
    ("Pancreas disease effects, type 1 diabetes", "pancreas_biology", "disease_logfc_spearman", "T1D_vs_Control", 1.0),
    ("Zebrafish dynamics", "trajectory/zebrafish", "dynamic_gene_spearman", None, 1.0),
    ("Zebrafish stage error", "trajectory/zebrafish", "stage_rank_mae", None, 1.0),
    ("Norman response edge AUPRC", "grn/norman_crispra", "edge_pr_auc_q10", None, 1.0),
)
SOURCES = {
    "safe_fusion": "Safe Fusion", "svd": "SVD", "weighted_knn": "Weighted kNN", "magic": "MAGIC", "scvi": "scVI",
    "dca": "DCA", "scimpute": "scImpute", "enimpute": "EnImpute",
}
NAME = re.compile(r"^(?P<source>.+?)_(?P<tail>\d+pct|dense)(?:__(?P<part>masked_only|zeros_only))?$")

def select(frame: pd.DataFrame, metric: str, contrast: str | None) -> pd.DataFrame:
    frame = frame[frame["metric"] == metric]
    if "contrast" in frame:
        frame = frame[frame["contrast"] == (contrast or "all_conditions")]
    return frame

def parse(method: str) -> tuple[str, str, str]:
    match = NAME.match(method)
    if not match:
        return method, "", "full"
    return match["source"], match["tail"], match["part"] or "full"

def comparisons(root: Path, endpoints=ENDPOINTS) -> pd.DataFrame:
    rows = []
    for label, subpath, metric, contrast, scale in endpoints:
        path = root / subpath / "paired_comparisons.parquet"
        if not path.exists():
            continue
        frame = select(pd.read_parquet(path), metric, contrast)
        frame = frame[frame["reference"] == "corrupted_raw"]
        for record in frame.itertuples():
            source, tail, part = parse(record.method)
            rows.append({
                "endpoint": label, "source": source, "method": SOURCES.get(source, source), "fill": tail, "part": part,
                "difference": record.difference * scale, "ci_low": record.ci_low * scale, "ci_high": record.ci_high * scale,
                "excludes_zero": bool(record.ci_low > 0 or record.ci_high < 0),
            })
    return pd.DataFrame(rows)

def estimates(root: Path) -> pd.DataFrame:
    rows = []
    for label, subpath, metric, contrast, scale in ENDPOINTS:
        path = root / subpath / "bootstrap_summary.parquet"
        if not path.exists():
            continue
        frame = select(pd.read_parquet(path), metric, contrast)
        for record in frame.itertuples():
            rows.append({"endpoint": label, "method": record.method, "estimate": record.estimate * scale,
                         "ci_low": record.ci_low * scale, "ci_high": record.ci_high * scale})
    return pd.DataFrame(rows)

def reproduction(root: Path, published_root: Path, overrides: dict, value: str) -> pd.DataFrame:
    rows = []
    for subpath in sorted({subpath for _, subpath, *_ in ENDPOINTS}):
        name = "paired_comparisons.parquet" if value == "difference" else "bootstrap_summary.parquet"
        new_path = root / subpath / name
        old_path = overrides.get(subpath, published_root / subpath) / name
        if not new_path.exists() or not old_path.exists():
            continue
        new, old = pd.read_parquet(new_path), pd.read_parquet(old_path)
        keys = [column for column in ("scope", "contrast", "method", "reference", "metric") if column in new and column in old]
        merged = new.merge(old, on=keys, suffixes=("_new", "_old"))
        merged = merged[merged["method"].str.match(r"^(safe_fusion|svd|weighted_knn)_\d+pct$")]
        rows.append({
            "evaluation": subpath, "value": value, "n_rows": int(len(merged)),
            "max_abs_difference": float((merged[f"{value}_new"] - merged[f"{value}_old"]).abs().max()) if len(merged) else float("nan"),
        })
    return pd.DataFrame(rows)

def formatted_table3(table: pd.DataFrame) -> pd.DataFrame:
    full = table[table["part"] == "full"]
    methods = [name for name in SOURCES.values() if name in set(full["method"])]
    rows = []
    for label, _, _, _, scale in ENDPOINTS:
        digits = 1 if scale == 100.0 else 3
        row = {"Endpoint": label}
        for method in methods:
            cells = []
            for pct in ("1pct", "5pct", "10pct"):
                hit = full[(full["endpoint"] == label) & (full["method"] == method) & (full["fill"] == pct)]
                if hit.empty:
                    cells.append("")
                    continue
                cells.append(f"{hit['difference'].iloc[0]:.{digits}f}{'*' if hit['excludes_zero'].iloc[0] else ''}")
            row[method] = " / ".join(cells)
        rows.append(row)
    return pd.DataFrame(rows)

def formatted_s17(values: pd.DataFrame) -> pd.DataFrame:
    columns = ["corrupted_raw", "reference_truth"]
    for source in SOURCES:
        columns += [f"{source}_{pct}pct" for pct in (1, 5, 10)]
    columns += [f"{source}_dense" for source in ("magic", "scvi")]
    pivot = values.pivot_table(index="endpoint", columns="method", values="estimate", aggfunc="first")
    pivot = pivot[[column for column in columns if column in pivot.columns]]
    return pivot.reindex([label for label, *_ in ENDPOINTS]).rename(columns={"corrupted_raw": "Masked", "reference_truth": "Unmasked"})

def decomposition(root: Path) -> pd.DataFrame:
    frames = []
    locations = {
        "safe_fusion": EVIDENCE / "downstream_decomposition", "svd": EVIDENCE / "downstream_decomposition",
        "weighted_knn": EVIDENCE / "downstream_decomposition",
        "magic": root / "masked" / "decomposition", "scvi": root / "masked" / "decomposition",
    }
    evaluations = {
        "colon": (("Colon marker AUPRC", "evaluation/markers", "canonical_marker_pr_auc", None, 100.0),
                  ("Colon adjusted Rand index", "evaluation/clustering", "annotation_ari", None, 1.0)),
        "pancreas": (("Pancreas marker AUPRC", "evaluation/biology", "canonical_marker_pr_auc", None, 100.0),
                     ("Pancreas adjusted Rand index", "evaluation/clustering/combined", "annotation_ari", None, 1.0),
                     ("Pancreas disease effects, autoantibody", "evaluation/biology", "disease_logfc_spearman", "AAB_vs_Control", 1.0),
                     ("Pancreas disease effects, type 1 diabetes", "evaluation/biology", "disease_logfc_spearman", "T1D_vs_Control", 1.0)),
        "zebrafish": (("Zebrafish dynamics", "evaluation/trajectory", "dynamic_gene_spearman", None, 1.0),
                      ("Zebrafish stage error", "evaluation/trajectory", "stage_rank_mae", None, 1.0)),
    }
    for source, base in locations.items():
        for dataset, endpoints in evaluations.items():
            unit = base / dataset / source
            if not unit.exists():
                continue
            table = comparisons(unit, endpoints)
            table = table[table["fill"].isin(["1pct", "5pct", "10pct"])]
            counts = pd.concat([pd.read_csv(path) for path in sorted(unit.rglob("decomposition_counts.csv"))])
            counts["fill"] = counts["source"].str.extract(r"_(\d+pct)$")[0]
            pooled = counts.groupby("fill")[["n_changed", "n_changed_masked"]].sum()
            table["masked_share"] = table["fill"].map(pooled["n_changed_masked"] / pooled["n_changed"])
            table["dataset"] = dataset
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def disease_effects(root: Path) -> pd.DataFrame:
    frames = []
    for tissue in ("pancreas", "colon"):
        for name in ("disease_effects", "disease_effects_comparators"):
            summary = root / "deployment" / name / tissue / "observed_summary.csv"
            null = root / "deployment" / name / tissue / "permutation_null.csv"
            if not summary.exists():
                continue
            observed = pd.read_csv(summary)
            permutations = pd.read_csv(null)
            merged = observed.merge(permutations, on=["method", "fraction", "contrast", "test"], how="left")
            frames.append(merged.assign(tissue=tissue))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round3/downstream")
    args = parser.parse_args()
    output = args.root / "summary"
    output.mkdir(parents=True, exist_ok=True)

    table3 = pd.concat([
        comparisons(args.root / "deployment" / "evaluation"),
        comparisons(args.root / "deployment" / "evaluation_comparators"),
    ], ignore_index=True)
    table3.to_csv(output / "table3.csv", index=False)
    formatted_table3(table3).to_csv(output / "table3_formatted.csv", index=False)

    comparator_values = estimates(args.root / "masked" / "evaluation_comparators")
    if len(comparator_values):
        comparator_values = comparator_values[~comparator_values["method"].isin(["corrupted_raw", "reference_truth"])]
    s17 = pd.concat([estimates(args.root / "masked" / "evaluation"), comparator_values], ignore_index=True)
    s17.to_csv(output / "s17.csv", index=False)
    formatted_s17(s17).round(4).to_csv(output / "s17_formatted.csv")
    pd.concat([
        comparisons(args.root / "masked" / "evaluation"),
        comparisons(args.root / "masked" / "evaluation_comparators"),
    ], ignore_index=True).to_csv(output / "s17_differences.csv", index=False)

    decomposition(args.root).to_csv(output / "decomposition.csv", index=False)
    disease_effects(args.root).to_csv(output / "disease_effects.csv", index=False)

    checks = pd.concat([
        reproduction(args.root / "deployment" / "evaluation", EVIDENCE / "downstream_deployment" / "evaluation",
                     PUBLISHED_DEPLOYMENT, "difference").assign(analysis="Table 3"),
        reproduction(args.root / "masked" / "evaluation", EVIDENCE / "downstream_complete",
                     PUBLISHED_MASKED, "estimate").assign(analysis="Table S17"),
    ], ignore_index=True)
    checks.to_csv(output / "reproduction.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(formatted_table3(table3).to_string(index=False))
        print(formatted_s17(s17).round(3).to_string())
        print(checks.to_string(index=False))

if __name__ == "__main__":
    main()
