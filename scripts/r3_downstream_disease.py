#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from disease_control_common import (
    DRAWS,
    FRACTIONS,
    SEED,
    TISSUES,
    fill_contract,
    interval,
    load_heldout,
    stratified_draws,
)
from disease_control_effects import EFFECT_STATISTICS, build_contrasts, evaluate, labelings

REPOSITORY = Path(__file__).resolve().parents[1]
TABLE3_METHODS = ("Safe Fusion", "SVD", "Weighted kNN")
STANDARD_METHODS = {"MAGIC": "magic", "scVI": "scvi"}
COMPARATOR_METHODS = {"DCA": "dca", "scImpute": "scimpute", "EnImpute": "enimpute"}

def unit_name(key: str) -> str:
    return f"pancreas/fold_{key.split('_')[1]}" if key.startswith("pancreas_") else key

def contract(key: str, method: str, fraction: float, root: Path) -> Path:
    percent = int(round(100 * fraction))
    if method in TABLE3_METHODS:
        return fill_contract(key, method, fraction)
    name = {**STANDARD_METHODS, **COMPARATOR_METHODS}[method]
    return root / "deployment" / unit_name(key) / f"{name}_{percent}pct"

def method_list(spec: dict, root: Path, comparators_only: bool) -> list[str]:
    available = [
        name for name in COMPARATOR_METHODS
        if all(contract(key, name, fraction, root).is_dir() for key in spec["units"] for fraction in FRACTIONS)
    ]
    return available if comparators_only else [*TABLE3_METHODS, *STANDARD_METHODS]

def filled(data, key_contract) -> np.ndarray:
    result = data.recorded.copy()
    for key, test in data.test_masks.items():
        path = key_contract(key)
        metadata = json.loads((path / "metadata.json").read_text())
        if metadata["gene_ids"] != data.gene_ids or metadata["cell_ids"] != data.obs["cell_id"].tolist():
            raise ValueError(f"cell or gene order differs for {path}")
        values = np.asarray(np.load(path / "mean.npy", mmap_mode="r")[test], dtype=np.float32)
        recorded = data.recorded[test]
        if not np.array_equal(values[recorded > 0], recorded[recorded > 0]):
            raise ValueError(f"{path} changes recorded nonzero counts")
        result[test] = values
    return result[data.heldout]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round3/downstream")
    parser.add_argument("--comparators-only", action="store_true")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()
    spec = TISSUES[args.tissue]

    data = load_heldout(args.tissue)
    recorded = data.counts
    contrasts = build_contrasts(data.cells, spec)
    draws = {c.name: stratified_draws(c.is_case, args.draws, args.seed + i) for i, c in enumerate(contrasts)}
    nulls = {c.name: labelings(c, args.permutations, args.seed + 100 + i) for i, c in enumerate(contrasts)}
    methods = method_list(spec, args.root, args.comparators_only)
    jobs = [(method, fraction) for method in methods for fraction in FRACTIONS]
    matrices = {
        job: filled(data, lambda key, job=job: contract(key, *job, args.root))
        for job in jobs
    }
    results = Parallel(n_jobs=args.jobs)(
        delayed(evaluate)(recorded, matrices[job], contrasts, draws, nulls) for job in jobs
    )

    summary_rows, null_rows = [], []
    for (method, fraction), result in zip(jobs, results):
        boot = pd.DataFrame(result["bootstrap"])
        for row in result["observed"]:
            record = {"method": method, "fraction": fraction, **row}
            samples = boot[(boot["contrast"] == row["contrast"]) & (boot["test"] == row["test"])]
            for name in EFFECT_STATISTICS:
                record[f"{name}_ci_low"], record[f"{name}_ci_high"] = interval(samples[name].to_numpy())
            summary_rows.append(record)
        null = pd.DataFrame(result["null"])
        for (name, test), frame in null.groupby(["contrast", "test"], sort=False):
            difference = (frame["n_discoveries_filled"] - frame["n_discoveries_unfilled"]).to_numpy()
            low, high = interval(difference)
            null_rows.append({
                "method": method, "fraction": fraction, "contrast": name, "test": test,
                "n_permutations": int(len(frame)),
                "false_discoveries_unfilled_mean": float(frame["n_discoveries_unfilled"].mean()),
                "false_discoveries_filled_mean": float(frame["n_discoveries_filled"].mean()),
                "false_discoveries_difference_mean": float(difference.mean()),
                "false_discoveries_difference_low": low,
                "false_discoveries_difference_high": high,
            })

    output = args.root / "deployment" / ("disease_effects_comparators" if args.comparators_only else "disease_effects") / args.tissue
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output / "observed_summary.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output / "permutation_null.csv", index=False)
    print(pd.DataFrame(summary_rows)[["method", "fraction", "contrast", "test", "n_discoveries_unfilled", "n_discoveries_filled", "effect_slope", "effect_slope_ci_low", "effect_slope_ci_high"]].to_string(index=False))

if __name__ == "__main__":
    main()
