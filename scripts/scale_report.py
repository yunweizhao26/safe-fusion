#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

STEPS = {
    "gene_median": ("Safe Fusion", "Gene median teacher"),
    "svd_impute": ("Safe Fusion", "SVD teacher"),
    "graph_smooth": ("Safe Fusion", "Weighted kNN teacher"),
    "magic_inductive": ("Safe Fusion", "MAGIC teacher"),
    "scvi_teacher": ("Safe Fusion", "scVI teacher"),
    "stack": ("Safe Fusion", "Fused value"),
    "selector": ("Safe Fusion", "Selector"),
    "magic": ("Comparison", "MAGIC"),
    "scvi": ("Comparison", "scVI"),
    "scvi_probability": ("Comparison", "scVI probability"),
    "alra": ("Comparison", "ALRA"),
    "saver": ("Comparison", "SAVER"),
    "transductive:gene_median": ("Transductive Safe Fusion", "Gene median teacher"),
    "transductive:svd_impute": ("Transductive Safe Fusion", "SVD teacher"),
    "transductive:graph_smooth": ("Transductive Safe Fusion", "Weighted kNN teacher"),
    "transductive:magic": ("Transductive Safe Fusion", "MAGIC teacher, count scale of standard MAGIC"),
    "transductive:scvi": ("Transductive Safe Fusion", "scVI teacher, count scale of standard scVI"),
    "transductive_stack": ("Transductive Safe Fusion", "Fused value"),
    "transductive_selector": ("Transductive Safe Fusion", "Selector"),
    "without_magic_stack": ("Safe Fusion without MAGIC teacher", "Fused value"),
    "without_magic_selector": ("Safe Fusion without MAGIC teacher", "Selector"),
    "stack_replicate": ("Timing replicate", "Fused value"),
    "transductive_stack_replicate": ("Timing replicate", "Fused value, transductive"),
    "without_magic_stack_replicate": ("Timing replicate", "Fused value without MAGIC teacher"),
    "mask": ("Data", "Mask"),
}
REPLICATE_OF = {
    "stack_replicate": "stack",
    "transductive_stack_replicate": "transductive_stack",
    "without_magic_stack_replicate": "without_magic_stack",
}
STORED_SELECTORS = {
    "pancreas_0": "artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit/fold_0/"
                  "selector_mlp_biology_range_fullteachers/calibration_report.json",
    "norman_crispra": "artifacts/paper_evidence/review_round2/leakage_free/selector_mlp_biology_range_fullteachers/"
                      "norman_crispra/calibration_report.json",
}

def seconds(value: str) -> float:
    if not value:
        return float("nan")
    days, _, clock = value.rpartition("-")
    parts = [float(part) for part in clock.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return 86400 * float(days or 0) + 3600 * parts[0] + 60 * parts[1] + parts[2]

def gigabytes(value: str) -> float:
    match = re.fullmatch(r"([\d.]+)([KMGT]?)", value or "")
    if not match:
        return float("nan")
    scale = {"": 1 / 1024**3, "K": 1 / 1024**2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}[match.group(2)]
    return float(match.group(1)) * scale

def sacct(jobs: list[str]) -> pd.DataFrame:
    fields = ["JobID", "NodeList", "State", "Elapsed", "TotalCPU", "MaxRSS", "AllocTRES", "TRESUsageInMax", "Timelimit"]
    output = subprocess.run(["sacct", "-P", "--noheader", "-j", ",".join(jobs), f"--format={','.join(fields)}"],
                            check=True, capture_output=True, text=True).stdout
    frame = pd.DataFrame([line.split("|") for line in output.strip().splitlines()], columns=fields)
    allocation = frame[~frame["JobID"].str.contains(r"\.")].set_index("JobID")
    batch = frame[frame["JobID"].str.endswith(".batch")].copy()
    batch["JobID"] = batch["JobID"].str.replace(".batch", "", regex=False)
    batch = batch.set_index("JobID")
    rows = []
    for job in jobs:
        if job not in allocation.index:
            rows.append({"job": job, "state": "UNKNOWN"})
            continue
        head = allocation.loc[job]
        step = batch.loc[job] if job in batch.index else head
        gpu = re.search(r"gres/gpumem=([\d.]+[KMGT]?)", step["TRESUsageInMax"])
        rows.append({
            "job": job, "node": head["NodeList"], "state": head["State"].split()[0],
            "time_limit_hours": seconds(head["Timelimit"]) / 3600,
            "wall_minutes": seconds(head["Elapsed"]) / 60, "cpu_hours": seconds(step["TotalCPU"]) / 3600,
            "cpus": int(re.search(r"cpu=(\d+)", head["AllocTRES"]).group(1)),
            "gpus": int(match.group(1)) if (match := re.search(r"gres/gpu=(\d+)", head["AllocTRES"])) else 0,
            "peak_memory_gb": gigabytes(step["MaxRSS"]),
            "peak_gpu_memory_gb": gigabytes(gpu.group(1)) if gpu else float("nan"),
        })
    return pd.DataFrame(rows)

def fits(resources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, step), frame in resources.groupby(["group", "step"], sort=False):
        done = frame[frame["state"] == "COMPLETED"]
        if len(done) < 2:
            continue
        cells = done["cells"].to_numpy(float)
        row = {"group": group, "step": step, "sizes": int(len(done))}
        for column in ("wall_minutes", "cpu_hours"):
            positive = done[column].to_numpy(float) > 0
            if positive.sum() >= 2:
                row[f"{column}_exponent"] = float(np.polyfit(np.log(cells[positive]), np.log(done[column].to_numpy(float)[positive]), 1)[0])
        slope, intercept = np.polyfit(cells, done["peak_memory_gb"].to_numpy(float), 1)
        row["peak_memory_gb_per_million_cells"] = float(1e6 * slope)
        row["peak_memory_gb_intercept"] = float(intercept)
        if done["peak_gpu_memory_gb"].notna().sum() >= 2:
            gpu = done.dropna(subset=["peak_gpu_memory_gb"])
            row["peak_gpu_memory_gb_per_million_cells"] = float(1e6 * np.polyfit(gpu["cells"].to_numpy(float), gpu["peak_gpu_memory_gb"].to_numpy(float), 1)[0])
        rows.append(row)
    return pd.DataFrame(rows)

def accuracy(root: Path) -> pd.DataFrame:

    summary = pd.read_csv(root / "results" / "method_summary.csv")
    differences = pd.read_csv(root / "results" / "paired_differences.csv")
    wide = summary.pivot_table(index=["cells", "method"], columns="statistic", values=["estimate", "lower", "upper"])
    wide.columns = [f"{statistic}_{bound}" for bound, statistic in wide.columns]
    paired = differences.pivot_table(index=["cells", "reference", "method"], columns="statistic",
                                     values=["difference", "lower", "upper"])
    paired.columns = [f"{statistic}_difference" + ("" if bound == "difference" else f"_{bound}")
                      for bound, statistic in paired.columns]
    table = paired.reset_index().merge(wide.reset_index(), on=["cells", "method"], how="left")
    numeric = [column for column in table.columns if column not in ("cells", "reference", "method")]
    table[numeric] = 100 * table[numeric]
    return table

def reproduction(root: Path) -> pd.DataFrame:
    rows = []
    for name, stored in STORED_SELECTORS.items():
        path = root / "checks" / "selector_reproduction" / name / "selector_report.json"
        if not path.exists():
            continue
        new, old = json.loads(path.read_text()), json.loads(Path(stored).read_text())
        equal = sum(a["n_true_positive"] == b["n_true_positive"] and a["n_selected"] == b["n_selected"]
                    for a, b in zip(new["exact_budget_curve"], old["exact_budget_curve"]))
        rows.append({
            "unit": name, "fit_candidates": new["fit_candidates"], "mlp_fit_rows_stored": old["models"]["mlp"]["fit_rows"],
            "mlp_fit_rows_scale_selector": new["models"]["mlp"]["fit_rows"],
            "test_ap_stored": old["test"]["pr_auc"], "test_ap_scale_selector": new["test"]["pr_auc"],
            "test_auroc_stored": old["test"]["roc_auc"], "test_auroc_scale_selector": new["test"]["roc_auc"],
            "mlp_iterations_stored": old["models"]["mlp"]["model_iterations"],
            "mlp_iterations_scale_selector": new["models"]["mlp"]["model_iterations"],
            "curve_points_identical": equal, "curve_points": len(old["exact_budget_curve"]),
        })
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/paper_evidence/review_round3/scale"))
    args = parser.parse_args()

    jobs = pd.read_csv(args.root / "jobs.tsv", sep="\t", dtype=str)
    jobs = jobs[jobs["step"].isin(STEPS)].drop_duplicates(["step", "cells"], keep="last")
    resources = jobs.merge(sacct(jobs["job"].tolist()), on="job")
    resources["cells"] = resources["cells"].astype(int)
    resources["group"] = resources["step"].map(lambda step: STEPS[step][0])
    resources["label"] = resources["step"].map(lambda step: STEPS[step][1])
    resources["order"] = resources["step"].map(list(STEPS).index)
    resources = resources.sort_values(["order", "cells"]).drop(columns="order")
    output = args.root / "results"
    output.mkdir(parents=True, exist_ok=True)
    resources.to_csv(output / "resources.csv", index=False)

    fastest = resources.assign(step=resources["step"].replace(REPLICATE_OF))
    fastest = fastest[fastest["state"] == "COMPLETED"].sort_values("wall_minutes").drop_duplicates(["step", "cells"])
    fastest["group"] = fastest["step"].map(lambda step: STEPS[step][0])
    fastest["order"] = fastest["step"].map(list(STEPS).index)
    fastest = fastest.sort_values(["order", "cells"]).drop(columns="order")
    fastest.to_csv(output / "resources_fastest_run.csv", index=False)
    scaling = fits(fastest)
    scaling.to_csv(output / "scaling_fits.csv", index=False)
    variants = {
        "Safe Fusion": fastest[fastest["group"] == "Safe Fusion"],
        "Transductive Safe Fusion, with standard MAGIC and scVI fits": fastest[
            (fastest["group"] == "Transductive Safe Fusion") | fastest["step"].isin(["magic", "scvi"])],
        "Safe Fusion without MAGIC teacher": fastest[
            (fastest["group"] == "Safe Fusion without MAGIC teacher")
            | fastest["step"].isin(["gene_median", "svd_impute", "graph_smooth", "scvi_teacher"])],
    }
    totals = pd.concat([frame.groupby("cells").agg(
        steps=("step", "size"), completed=("state", lambda states: int((states == "COMPLETED").sum())),
        wall_minutes_sum=("wall_minutes", "sum"), cpu_hours_sum=("cpu_hours", "sum"),
        peak_memory_gb_max=("peak_memory_gb", "max"), peak_gpu_memory_gb_max=("peak_gpu_memory_gb", "max"),
    ).reset_index().assign(variant=name) for name, frame in variants.items()], ignore_index=True)
    totals.to_csv(output / "safe_fusion_totals.csv", index=False)
    reproduction(args.root).to_csv(output / "selector_reproduction.csv", index=False)
    if (output / "paired_differences.csv").exists():
        accuracy(args.root).to_csv(output / "accuracy_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(resources[["group", "label", "cells", "state", "wall_minutes", "cpu_hours", "peak_memory_gb", "peak_gpu_memory_gb"]].round(3).to_string(index=False))
        print(scaling.round(3).to_string(index=False))
        print(totals.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
