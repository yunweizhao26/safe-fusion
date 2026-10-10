#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from disease_control_common import (
    DRAWS,
    FRACTIONS,
    METHODS,
    SEED,
    TISSUES,
    filled_matrix,
    interval,
    load_heldout,
    output_dir,
    stratified_draws,
)

FDR = 0.05
TESTS = ("pseudobulk", "cell_level")
EFFECT_STATISTICS = ("sign_change_share", "effect_slope")


@dataclass
class Contrast:
    name: str
    donors: list[str]
    is_case: np.ndarray
    cell_types: list[str]
    cells: dict[tuple[int, int], np.ndarray]


def build_contrasts(cells: pd.DataFrame, spec: dict) -> list[Contrast]:
    contrasts = []
    for case in spec["cases"]:
        keep = cells["condition_label"].isin([case, spec["control"]])
        donors = sorted(cells.loc[keep, "donor"].unique())
        condition = cells.groupby("donor")["condition_label"].first().loc[donors].to_numpy()
        cell_types = sorted(cells.loc[keep, "cell_type"].unique())
        groups = {
            (d, c): np.flatnonzero((cells["donor"].to_numpy() == donor) & (cells["cell_type"].to_numpy() == cell_type))
            for d, donor in enumerate(donors)
            for c, cell_type in enumerate(cell_types)
        }
        contrasts.append(Contrast(
            name=f"{case} vs {spec['control']}",
            donors=donors,
            is_case=condition == case,
            cell_types=cell_types,
            cells={key: rows for key, rows in groups.items() if len(rows)},
        ))
    return contrasts


def prepared(matrix: np.ndarray, contrast: Contrast) -> dict[str, np.ndarray]:
    library = matrix.sum(axis=1, dtype=np.float64)
    normalized = np.log1p(matrix * np.divide(1e4, library, out=np.zeros_like(library), where=library > 0)[:, None]).astype(np.float32)
    cube = np.full((len(contrast.donors), len(contrast.cell_types), matrix.shape[1]), np.nan, dtype=np.float32)
    for (d, c), rows in contrast.cells.items():
        total = matrix[rows].sum(axis=0, dtype=np.float64)
        if total.sum() > 0:
            cube[d, c] = np.log1p(total * (1e6 / total.sum()))
    return {"normalized": normalized, "cube": cube}


def run_tests(values: dict[str, np.ndarray], contrast: Contrast, case: np.ndarray, control: np.ndarray, with_tests: bool = True) -> dict[str, tuple]:
    n_types, n_genes = len(contrast.cell_types), values["normalized"].shape[1]
    result = {}
    for test in TESTS:
        effect = np.zeros((n_types, n_genes), dtype=np.float32)
        pvalue = np.ones((n_types, n_genes), dtype=np.float64)
        qvalue = np.ones((n_types, n_genes), dtype=np.float64)
        tested = np.zeros(n_types, dtype=bool)
        for c in range(n_types):
            case_donors = [d for d in case if (d, c) in contrast.cells]
            control_donors = [d for d in control if (d, c) in contrast.cells]
            if len(case_donors) < 2 or len(control_donors) < 2:
                continue
            if test == "pseudobulk":
                a = values["cube"][case_donors, c]
                b = values["cube"][control_donors, c]
            else:
                a = values["normalized"][np.concatenate([contrast.cells[(d, c)] for d in case_donors])]
                b = values["normalized"][np.concatenate([contrast.cells[(d, c)] for d in control_donors])]
            effect[c] = a.mean(axis=0) - b.mean(axis=0)
            tested[c] = True
            if not with_tests:
                continue
            if test == "pseudobulk":
                p = stats.ttest_ind(a, b, axis=0, equal_var=False).pvalue
            else:
                p = stats.mannwhitneyu(a, b, axis=0, method="asymptotic").pvalue
            p = np.nan_to_num(p, nan=1.0)
            pvalue[c] = p
            qvalue[c] = stats.false_discovery_control(p)
        result[test] = (effect, pvalue, qvalue, tested)
    return result


def compare(raw: tuple, filled: tuple) -> dict[str, int]:
    raw_effect, _, raw_q, tested = raw
    filled_effect, _, filled_q, _ = filled
    raw_sig = (raw_q < FDR) & tested[:, None]
    filled_sig = (filled_q < FDR) & tested[:, None]
    flip = (raw_effect * filled_effect < 0) & tested[:, None]
    pairs = np.broadcast_to(tested[:, None], raw_effect.shape)
    x, y = raw_effect[pairs].astype(np.float64), filled_effect[pairs].astype(np.float64)
    return {
        "n_tested_pairs": int(tested.sum() * raw_effect.shape[1]),
        "n_discoveries_unfilled": int(raw_sig.sum()),
        "n_discoveries_filled": int(filled_sig.sum()),
        "n_gained": int((filled_sig & ~raw_sig).sum()),
        "n_lost": int((raw_sig & ~filled_sig).sum()),
        "n_sign_changes": int(flip.sum()),
        "n_sign_changes_among_discoveries": int((flip & (raw_sig | filled_sig)).sum()),
        "sign_change_share": float(flip.sum() / max(1, int(((raw_effect != 0) & tested[:, None]).sum()))),
        "effect_slope": float(np.dot(x, y) / np.dot(x, x)) if np.dot(x, x) > 0 else float("nan"),
    }


def labelings(contrast: Contrast, permutations: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(contrast.donors)
    k = int(contrast.is_case.sum())
    observed = tuple(np.flatnonzero(contrast.is_case))
    if math.comb(n, k) - 1 <= permutations:
        chosen = [combo for combo in itertools.combinations(range(n), k) if combo != observed]
    else:
        rng = np.random.default_rng(seed)
        seen = {observed}
        chosen = []
        while len(chosen) < permutations:
            combo = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
            if combo not in seen:
                seen.add(combo)
                chosen.append(combo)
    everyone = np.arange(n)
    return [(np.asarray(combo), np.setdiff1d(everyone, combo)) for combo in chosen]


def evaluate(recorded: np.ndarray, matrix: np.ndarray, contrasts: list[Contrast], draws: dict, nulls: dict) -> dict:
    rows, boot_rows, null_rows, effects = [], [], [], {}
    for contrast in contrasts:
        raw_values = prepared(recorded, contrast)
        filled_values = prepared(matrix, contrast)
        observed_case = np.flatnonzero(contrast.is_case)
        observed_control = np.flatnonzero(~contrast.is_case)
        raw = run_tests(raw_values, contrast, observed_case, observed_control)
        filled = run_tests(filled_values, contrast, observed_case, observed_control)
        effects[contrast.name] = (raw, filled)
        for test in TESTS:
            rows.append({"contrast": contrast.name, "test": test, **compare(raw[test], filled[test])})
        for index, draw in enumerate(draws[contrast.name]):
            case = draw[contrast.is_case[draw]]
            control = draw[~contrast.is_case[draw]]
            raw = run_tests(raw_values, contrast, case, control, with_tests=False)
            filled = run_tests(filled_values, contrast, case, control, with_tests=False)
            for test in TESTS:
                counts = compare(raw[test], filled[test])
                boot_rows.append({"contrast": contrast.name, "test": test, "draw": index,
                                  **{name: counts[name] for name in EFFECT_STATISTICS}})
        for index, (case, control) in enumerate(nulls[contrast.name]):
            raw = run_tests(raw_values, contrast, case, control)
            filled = run_tests(filled_values, contrast, case, control)
            for test in TESTS:
                null_rows.append({"contrast": contrast.name, "test": test, "permutation": index, **compare(raw[test], filled[test])})
    return {"observed": rows, "bootstrap": boot_rows, "null": null_rows, "effects": effects}


def effect_frame(effects: dict, contrasts: list[Contrast], symbols: np.ndarray, method: str, fraction: float, which: int) -> pd.DataFrame:
    frames = []
    for contrast in contrasts:
        for test in TESTS:
            effect, pvalue, qvalue, tested = effects[contrast.name][which][test]
            for c in np.flatnonzero(tested):
                frames.append(pd.DataFrame({
                    "contrast": contrast.name, "test": test, "method": method, "fraction": fraction,
                    "cell_type": contrast.cell_types[c], "gene": symbols,
                    "effect": effect[c], "p_value": pvalue[c], "fdr": qvalue[c],
                }))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = parser.parse_args()
    spec = TISSUES[args.tissue]
    output = output_dir("disease_effects", args.tissue)

    data = load_heldout(args.tissue)
    cells = data.cells
    recorded = data.counts
    contrasts = build_contrasts(cells, spec)
    draws = {c.name: stratified_draws(c.is_case, args.draws, args.seed + i) for i, c in enumerate(contrasts)}
    nulls = {c.name: labelings(c, args.permutations, args.seed + 100 + i) for i, c in enumerate(contrasts)}
    jobs = [(method, fraction) for method in METHODS for fraction in FRACTIONS]
    matrices = {job: filled_matrix(data, *job)[data.heldout] for job in jobs}
    results = Parallel(n_jobs=args.jobs)(
        delayed(evaluate)(recorded, matrices[job], contrasts, draws, nulls) for job in jobs
    )

    effect_frames, summary_rows, null_rows, null_counts = [], [], [], []
    for index, ((method, fraction), result) in enumerate(zip(jobs, results)):
        if index == 0:
            effect_frames.append(effect_frame(result["effects"], contrasts, data.symbols, "unfilled", 0.0, 0))
        effect_frames.append(effect_frame(result["effects"], contrasts, data.symbols, method, fraction, 1))
        boot = pd.DataFrame(result["bootstrap"])
        for row in result["observed"]:
            record = {"method": method, "fraction": fraction, **row}
            samples = boot[(boot["contrast"] == row["contrast"]) & (boot["test"] == row["test"])]
            for name in EFFECT_STATISTICS:
                record[f"{name}_ci_low"], record[f"{name}_ci_high"] = interval(samples[name].to_numpy())
            record["discovery_difference"] = row["n_discoveries_filled"] - row["n_discoveries_unfilled"]
            summary_rows.append(record)
        null = pd.DataFrame(result["null"]).assign(method=method, fraction=fraction)
        null_counts.append(null)
        for (contrast, test), frame in null.groupby(["contrast", "test"], sort=False):
            difference = (frame["n_discoveries_filled"] - frame["n_discoveries_unfilled"]).to_numpy()
            low, high = interval(difference)
            observed = next(r for r in result["observed"] if r["contrast"] == contrast and r["test"] == test)
            null_rows.append({
                "method": method, "fraction": fraction, "contrast": contrast, "test": test,
                "n_permutations": int(len(frame)),
                "n_tested_pairs_mean": float(frame["n_tested_pairs"].mean()),
                "false_discoveries_unfilled_mean": float(frame["n_discoveries_unfilled"].mean()),
                "false_discoveries_filled_mean": float(frame["n_discoveries_filled"].mean()),
                "false_discoveries_difference_mean": float(difference.mean()),
                "false_discoveries_difference_low": low,
                "false_discoveries_difference_high": high,
                "share_with_any_unfilled": float((frame["n_discoveries_unfilled"] > 0).mean()),
                "share_with_any_filled": float((frame["n_discoveries_filled"] > 0).mean()),
                "false_gained_mean": float(frame["n_gained"].mean()),
                "observed_discoveries_unfilled": observed["n_discoveries_unfilled"],
                "observed_discoveries_filled": observed["n_discoveries_filled"],
                "permutation_p_unfilled": float((1 + (frame["n_discoveries_unfilled"] >= observed["n_discoveries_unfilled"]).sum()) / (1 + len(frame))),
                "permutation_p_filled": float((1 + (frame["n_discoveries_filled"] >= observed["n_discoveries_filled"]).sum()) / (1 + len(frame))),
            })

    pd.concat(effect_frames, ignore_index=True).to_parquet(output / "observed_effects.parquet", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "observed_summary.csv", index=False)
    pd.DataFrame(null_rows).to_csv(output / "permutation_null.csv", index=False)
    pd.concat(null_counts, ignore_index=True).to_parquet(output / "permutation_counts.parquet", index=False)
    print(pd.DataFrame(summary_rows)[["method", "fraction", "contrast", "test", "n_tested_pairs", "n_discoveries_unfilled", "n_discoveries_filled", "n_gained", "n_lost", "n_sign_changes"]].to_string(index=False))
    print(pd.DataFrame(null_rows)[["method", "fraction", "contrast", "test", "n_permutations", "false_discoveries_unfilled_mean", "false_discoveries_filled_mean", "share_with_any_unfilled", "share_with_any_filled"]].to_string(index=False))


if __name__ == "__main__":
    main()
