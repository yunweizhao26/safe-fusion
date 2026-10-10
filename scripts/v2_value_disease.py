#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "scripts"))

from disease_control_common import DRAWS, FRACTIONS, SEED, TISSUES, fill_contract, interval, load_heldout, stratified_draws
from disease_control_effects import EFFECT_STATISTICS, build_contrasts, evaluate, labelings

VALUE_ROOT = REPOSITORY / "artifacts" / "paper_evidence" / "review_round3" / "value_v2_ablations" / "value"
CURRENT = "Safe Fusion"

def unit_name(key: str) -> str:
    return f"pancreas/fold_{key.split('_')[1]}" if key.startswith("pancreas_") else key

def filled(data, contract_of) -> np.ndarray:
    result = data.recorded.copy()
    for key, test in data.test_masks.items():
        path = contract_of(key)
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--fills-root", type=Path, default=VALUE_ROOT / "deployment" / "fills")
    parser.add_argument("--values", nargs="+", required=True, help="Inserted values of the new fills, for example rate_poisson.")
    parser.add_argument("--output-dir", type=Path, default=VALUE_ROOT / "deployment" / "disease_effects")
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

    def contract(key: str, method: str, fraction: float) -> Path:
        if method == CURRENT:
            return fill_contract(key, CURRENT, fraction)
        return args.fills_root / unit_name(key) / f"{method}_{int(round(100 * fraction))}pct"

    alternatives = [f"safe_fusion_{value}" for value in args.values]
    methods = [CURRENT, *alternatives]
    jobs = [(method, fraction) for method in methods for fraction in FRACTIONS]
    matrices = {job: filled(data, lambda key, job=job: contract(key, *job)) for job in jobs}
    results = Parallel(n_jobs=args.jobs)(delayed(evaluate)(recorded, matrices[job], contrasts, draws, nulls) for job in jobs)

    summary_rows, null_rows, boots = [], [], {}
    for (method, fraction), result in zip(jobs, results):
        boot = pd.DataFrame(result["bootstrap"])
        boots[(method, fraction)] = boot
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
                "false_gained_mean": float(frame["n_gained"].mean()),
            })

    observed = pd.DataFrame(summary_rows)
    difference_rows = []
    point = observed.set_index(["method", "fraction", "contrast", "test"])["effect_slope"]
    for method in alternatives:
        for fraction in FRACTIONS:
            new, current = boots[(method, fraction)], boots[(CURRENT, fraction)]
            merged = new.merge(current, on=["contrast", "test", "draw"], suffixes=("_new", "_current"))
            for (name, test), frame in merged.groupby(["contrast", "test"], sort=False):
                low, high = interval((frame["effect_slope_new"] - frame["effect_slope_current"]).to_numpy())
                difference_rows.append({
                    "method": method, "fraction": fraction, "contrast": name, "test": test,
                    "slope_current": float(point[(CURRENT, fraction, name, test)]),
                    "slope_new": float(point[(method, fraction, name, test)]),
                    "difference": float(point[(method, fraction, name, test)] - point[(CURRENT, fraction, name, test)]),
                    "difference_low": low, "difference_high": high,
                })
    output = args.output_dir / args.tissue
    output.mkdir(parents=True, exist_ok=True)
    observed.to_csv(output / "observed_summary.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output / "permutation_null.csv", index=False)
    pd.DataFrame(difference_rows).to_csv(output / "slope_difference.csv", index=False)
    print(observed[["method", "fraction", "contrast", "test", "n_discoveries_unfilled", "n_discoveries_filled",
                    "effect_slope", "effect_slope_ci_low", "effect_slope_ci_high"]].to_string(index=False))
    print(pd.DataFrame(null_rows).to_string(index=False))
    print(pd.DataFrame(difference_rows).to_string(index=False))

if __name__ == "__main__":
    main()
