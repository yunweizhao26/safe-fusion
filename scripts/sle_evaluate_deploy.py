#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from sle_common import (
    CASE, FOLDS, METHODS, REPORTED, SEED, Unit, bootstrap_draws, config, filled_matrix, interval, load_fills, load_unit,
)

ALPHA = 0.05
GROUPS = {"Treg": ("lineage", "Treg"), "Classical monocyte": ("cell_type", "cM"), "All cells": (None, None)}
NULL_GROUPS = ("Treg", "Classical monocyte")

@dataclass
class Fold:
    unit: Unit
    fills: dict
    test_rows: np.ndarray
    common: np.ndarray

def variants() -> list[tuple]:
    items = [("Unfilled", None, None)]
    for mode in ("global", "donor", "gene"):
        for method in METHODS:
            for level in REPORTED:
                items.append((method, mode, level))
    return items

def pooled(folds: list[Fold], method: str, mode: str | None, level: int | None) -> np.ndarray:
    parts = []
    for fold in folds:
        matrix = fold.unit.counts[fold.test_rows] if mode is None else filled_matrix(fold.unit, fold.fills[method], level, mode)
        parts.append(matrix[:, fold.common])
    return np.vstack(parts)

def spearman_rows(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.full(len(weights), np.nan)
    for index, w in enumerate(weights):
        repeat = w.astype(int)
        xs, ys = np.repeat(x, repeat), np.repeat(y, repeat)
        if np.unique(xs).size > 1 and np.unique(ys).size > 1:
            values[index] = stats.spearmanr(xs, ys).statistic
    return values

def bh_count(p: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    m = p.shape[1]
    ordered = np.sort(p, axis=1)
    below = ordered <= alpha * np.arange(1, m + 1) / m
    return np.where(below.any(axis=1), m - np.argmax(below[:, ::-1], axis=1), 0)

def wilcoxon_p(values: np.ndarray, membership: np.ndarray) -> np.ndarray:

    n = values.shape[0]
    ranks = stats.rankdata(values, axis=0)
    ties = np.zeros(values.shape[1])
    for gene in range(values.shape[1]):
        _, counts = np.unique(values[:, gene], return_counts=True)
        ties[gene] = np.sum(counts.astype(float) ** 3 - counts)
    n1 = membership.sum(axis=1, keepdims=True)
    n2 = n - n1
    u = membership @ ranks - n1 * (n1 + 1) / 2
    variance = n1 * n2 / 12 * ((n + 1) - ties[None, :] / (n * (n - 1)))
    z = np.divide(u - n1 * n2 / 2, np.sqrt(np.maximum(variance, 0)), out=np.zeros_like(u), where=variance > 0)
    return np.where(variance > 0, 2 * stats.norm.sf(np.abs(z)), 1.0)

def welch_p(y: np.ndarray, membership: np.ndarray) -> np.ndarray:

    a = membership.astype(float)
    b = 1.0 - a
    stats_by_group = []
    for m in (a, b):
        n = m.sum(axis=1, keepdims=True)
        mean = m @ y / n
        var = (m @ y**2 - n * mean**2) / (n - 1)
        stats_by_group.append((n, mean, np.maximum(var, 0)))
    (n1, m1, v1), (n0, m0, v0) = stats_by_group
    se2 = v1 / n1 + v0 / n0
    t = np.divide(m1 - m0, np.sqrt(se2), out=np.zeros_like(se2), where=se2 > 0)
    df = np.divide(se2**2, (v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1), out=np.ones_like(se2),
                   where=se2 > 0)
    return np.where(se2 > 0, 2 * stats.t.sf(np.abs(t), df), 1.0)

def pseudobulk(matrix: np.ndarray, donor_index: np.ndarray, n_donors: int, cells: np.ndarray) -> np.ndarray:
    sums = np.zeros((n_donors, matrix.shape[1]))
    np.add.at(sums, donor_index[cells], matrix[cells])
    library = sums.sum(axis=1, keepdims=True)
    return np.log2(1e6 * np.divide(sums, library, out=np.zeros_like(sums), where=library > 0) + 1)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", default="main")
    parser.add_argument("--output-dir", type=Path, default=CASE / "results" / "deploy_main")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    settings = config()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    folds, common = [], None
    for fold in range(FOLDS):
        unit = load_unit(args.set_name, fold, "deploy")
        common = [gene for gene in (unit.genes if common is None else common) if gene in set(unit.genes)]
        folds.append(Fold(unit, {method: load_fills(unit, method) for method in METHODS}, np.flatnonzero(unit.test), None))
    for fold in folds:
        fold.common = np.asarray([fold.unit.genes.index(gene) for gene in common])
    obs = pd.concat([fold.unit.obs.iloc[fold.test_rows] for fold in folds])
    donor_condition = obs.drop_duplicates("donor").set_index("donor")["condition"].sort_index()
    donors = donor_condition.index.to_numpy()
    condition = donor_condition.to_numpy()
    sle = condition == "SLE"
    donor_index = pd.Index(donors).get_indexer(obs["donor"])
    lineage = obs["lineage"].astype(str).to_numpy()
    treg = lineage == "Treg"
    weights = np.vstack([np.ones(len(donors)), bootstrap_draws(donors, condition, args.draws, args.seed)])
    gene_position = {gene: index for index, gene in enumerate(common)}
    raw = pooled(folds, "Unfilled", None, None)
    summary = {"set": args.set_name, "donors": {"SLE": int(sle.sum()), "healthy": int((~sle).sum())},
               "cells": int(len(obs)), "tregs": int(treg.sum()), "shared_genes": len(common)}

    def by_condition(numerator: np.ndarray, denominator: np.ndarray) -> dict:
        result = {}
        rates = {}
        for label, members in (("SLE", sle), ("healthy", ~sle)):
            w = weights * members
            rates[label] = (w @ numerator) / (w @ denominator)
            result[label] = float(rates[label][0])
            result[f"{label}_ci"] = interval(rates[label][1:])
        difference = rates["SLE"] - rates["healthy"]
        result["SLE_minus_healthy"] = float(difference[0])
        result["SLE_minus_healthy_ci"] = interval(difference[1:])
        return result

    q4_library = []
    panel_zero = np.zeros(len(obs))
    panel_library = np.zeros(len(obs))
    offset = 0
    for fold in folds:
        counts = fold.unit.counts[fold.test_rows]
        panel_zero[offset:offset + len(counts)] = (counts == 0).mean(axis=1)
        panel_library[offset:offset + len(counts)] = counts.sum(axis=1)
        offset += len(counts)
    for group, members in (("All cells", np.ones(len(obs), bool)), ("Treg", treg)):
        cells_per_donor = np.bincount(donor_index[members], minlength=len(donors)).astype(float)
        for name, values in (("total UMI (all genes)", obs["total_counts"].to_numpy(float)),
                             ("UMI in fold panel", panel_library), ("zero fraction in fold panel", panel_zero)):
            per_donor = np.bincount(donor_index[members], weights=values[members], minlength=len(donors))
            donor_mean = per_donor / np.maximum(cells_per_donor, 1)
            q4_library.append({"cells": group, "quantity": name,
                               **by_condition(donor_mean, np.ones(len(donors)))})
    pd.DataFrame(q4_library).to_csv(output / "q4_library_zero_fraction.csv", index=False)

    q4_fill = []
    candidate_donor, candidate_treg = [], []
    for fold in folds:
        rows = fold.fills["Safe Fusion"]["rows"]
        candidate_donor.append(pd.Index(donors).get_indexer(fold.unit.obs["donor"].to_numpy())[rows])
        candidate_treg.append((fold.unit.obs["lineage"].to_numpy() == "Treg")[rows])
    for method in METHODS:
        for mode in ("global", "donor", "gene"):
            for level in REPORTED:
                filled_by_fold = [(fold.fills[method][f"first_{mode}"] <= level) & (fold.fills[method]["value"] > 0) for fold in folds]
                for scope, gene in (("all genes", None), *[(g, g) for g in settings["focus"]]):
                    for cells_name in ("All cells", "Treg"):
                        numerator = np.zeros(len(donors))
                        denominator = np.zeros(len(donors))
                        for index, fold in enumerate(folds):
                            keep = np.ones(len(candidate_donor[index]), bool) if gene is None else (
                                fold.fills[method]["cols"] == fold.unit.genes.index(gene))
                            if cells_name == "Treg":
                                keep = keep & candidate_treg[index]
                            d = candidate_donor[index][keep]
                            denominator += np.bincount(d, minlength=len(donors))
                            numerator += np.bincount(d, weights=filled_by_fold[index][keep].astype(float), minlength=len(donors))
                        q4_fill.append({"method": method, "budget": mode, "fill_pct": level, "genes": scope,
                                        "cells": cells_name, **by_condition(numerator, denominator)})
    pd.DataFrame(q4_fill).to_csv(output / "q4_fill_rates.csv", index=False)

    group_cells = {name: (np.ones(len(obs), bool) if column is None else obs[column].astype(str).to_numpy() == value)
                   for name, (column, value) in GROUPS.items()}
    rng = np.random.default_rng(args.seed)
    permutations = np.asarray([rng.permutation(sle) for _ in range(args.permutations)])
    membership = {"observed": sle[None, :], "null": permutations}
    raw_bulk = {name: pseudobulk(raw, donor_index, len(donors), cells) for name, cells in group_cells.items()}
    raw_null = {}

    def cell_log(matrix: np.ndarray) -> np.ndarray:
        library = matrix.sum(axis=1, keepdims=True)
        return np.log1p(1e4 * np.divide(matrix, library, out=np.zeros_like(matrix), where=library > 0))

    for group in NULL_GROUPS:
        cells = group_cells[group]
        values = cell_log(raw[cells])
        raw_null[group] = {
            kind: {"wilcoxon": bh_count(wilcoxon_p(values, m[:, donor_index[cells]].astype(float))),
                   "pseudobulk": bh_count(welch_p(raw_bulk[group], m))}
            for kind, m in membership.items()
        }

    split_rng = np.random.default_rng(args.seed)
    half = np.zeros(len(obs), bool)
    for d in range(len(donors)):
        members = np.flatnonzero(treg & (donor_index == d))
        half[split_rng.permutation(members)[: len(members) // 2]] = True

    import anndata as ad
    import scanpy as sc

    def module_scores(matrix: np.ndarray) -> dict[str, np.ndarray]:
        adata = ad.AnnData(X=cell_log(matrix).astype(np.float32))
        adata.var_names = common
        result = {}
        for name, genes in (("interferon", settings["interferon"]), ("treg_suppressive", settings["treg_suppressive"])):
            present = [gene for gene in genes if gene in gene_position]
            sc.tl.score_genes(adata, present, ctrl_size=50, n_bins=25, random_state=0, score_name=name)
            result[name] = adata.obs[name].to_numpy()
        return result

    raw_scores = module_scores(raw)
    q2_positive, q2_share, q2_conc, q4_effects, q4_null, q5_rows, q6_rows = [], [], [], [], [], [], []
    for method, mode, level in variants():
        if mode is None:
            continue
        matrix = pooled(folds, method, mode, level)
        label = {"method": method, "budget": mode, "fill_pct": level}
        for gene in settings["focus"]:
            g = gene_position[gene]
            before, after = raw[:, g] > 0, matrix[:, g] > 0
            n_treg = np.bincount(donor_index[treg], minlength=len(donors)).astype(float)
            pos_before = np.bincount(donor_index[treg & before], minlength=len(donors)).astype(float)
            pos_after = np.bincount(donor_index[treg & after], minlength=len(donors)).astype(float)
            row = {**label, "gene": gene}
            for name, numerator in (("before", pos_before), ("after", pos_after)):
                for cond, members in (("SLE", sle), ("healthy", ~sle)):
                    w = weights * members
                    fraction = (w @ numerator) / (w @ n_treg)
                    row[f"tregs_positive_{name}_{cond}"] = int(numerator[members].sum())
                    row[f"fraction_{name}_{cond}"] = float(fraction[0])
                    row[f"fraction_{name}_{cond}_ci"] = interval(fraction[1:])
            increase = by_condition(pos_after - pos_before, n_treg)
            row.update({f"increase_{key}": value for key, value in increase.items()})
            row["n_tregs_SLE"] = int(n_treg[sle].sum())
            row["n_tregs_healthy"] = int(n_treg[~sle].sum())
            q2_positive.append(row)

            filled = (raw[:, g] == 0) & (matrix[:, g] > 0)
            total = np.bincount(donor_index[filled], minlength=len(donors)).astype(float)
            zero = raw[:, g] == 0
            for lin in sorted(set(lineage)):
                in_lineage = lineage == lin
                fills_l = np.bincount(donor_index[filled & in_lineage], minlength=len(donors)).astype(float)
                share = np.divide(weights @ fills_l, weights @ total, out=np.full(len(weights), np.nan), where=(weights @ total) > 0)
                zero_share = zero[in_lineage].sum() / zero.sum()
                detect_share = before[in_lineage].sum() / max(before.sum(), 1)
                q2_share.append({**label, "gene": gene, "lineage": lin, "fills": int(fills_l.sum()),
                                 "share_of_fills": float(share[0]), "share_of_fills_ci": interval(share[1:]),
                                 "share_of_candidate_zeros": float(zero_share), "share_of_recorded_detections": float(detect_share),
                                 "cells_in_lineage": int(in_lineage.sum())})

            treg_zero = treg & zero
            n_zero = np.bincount(donor_index[treg_zero], minlength=len(donors)).astype(float)
            n_fill = np.bincount(donor_index[treg_zero & filled], minlength=len(donors)).astype(float)
            fill_rate = np.divide(n_fill, n_zero, out=np.zeros_like(n_fill), where=n_zero > 0)
            sums = np.bincount(donor_index[treg], weights=raw[treg, g], minlength=len(donors))
            library = np.bincount(donor_index[treg], weights=raw[treg].sum(axis=1), minlength=len(donors))
            raw_cp10k = 1e4 * sums / library
            rho = spearman_rows(raw_cp10k, fill_rate, weights)
            q2_conc.append({**label, "gene": gene, "spearman_raw_treg_expression_vs_treg_fill_rate": rho[0],
                            "ci": interval(rho[1:]), "treg_fill_rate_mean": float(fill_rate.mean()),
                            "donors_with_raw_treg_expression": int((raw_cp10k > 0).sum())})

            filled_sums = np.bincount(donor_index[treg], weights=matrix[treg, g], minlength=len(donors))
            filled_library = np.bincount(donor_index[treg], weights=matrix[treg].sum(axis=1), minlength=len(donors))
            filled_cp10k = 1e4 * filled_sums / filled_library
            rho = spearman_rows(raw_cp10k, filled_cp10k, weights)
            agree = (np.sign(raw_cp10k - np.median(raw_cp10k)) == np.sign(filled_cp10k - np.median(filled_cp10k)))
            agree_draws = (weights @ agree) / weights.sum(axis=1)
            halves = {}
            for name, members in (("A", treg & half), ("B", treg & ~half)):
                lib_raw = np.bincount(donor_index[members], weights=raw[members].sum(axis=1), minlength=len(donors))
                lib_fill = np.bincount(donor_index[members], weights=matrix[members].sum(axis=1), minlength=len(donors))
                halves[f"raw_{name}"] = 1e4 * np.bincount(donor_index[members], weights=raw[members, g], minlength=len(donors)) / lib_raw
                halves[f"filled_{name}"] = 1e4 * np.bincount(donor_index[members], weights=matrix[members, g], minlength=len(donors)) / lib_fill
            added = halves["filled_A"] - halves["raw_A"]
            split = {name: spearman_rows(x, halves["raw_B"], weights)
                     for name, x in (("raw_A_vs_raw_B", halves["raw_A"]), ("filled_A_vs_raw_B", halves["filled_A"]),
                                     ("added_A_vs_raw_B", added))}
            q6_rows.append({**label, "gene": gene, "spearman_raw_vs_filled": rho[0], "spearman_raw_vs_filled_ci": interval(rho[1:]),
                            "direction_agreement": float(agree_draws[0]), "direction_agreement_ci": interval(agree_draws[1:]),
                            **{f"split_half_{name}": float(v[0]) for name, v in split.items()},
                            **{f"split_half_{name}_ci": interval(v[1:]) for name, v in split.items()},
                            "donors": len(donors), "donors_with_zero_raw_treg_expression": int((raw_cp10k == 0).sum())})

        if mode == "global":
            scores = module_scores(matrix)
            for module, cells_name, cells in (("interferon", "All cells", np.ones(len(obs), bool)), ("interferon", "Treg", treg),
                                              ("treg_suppressive", "Treg", treg)):
                r = float(np.corrcoef(raw_scores[module][cells], scores[module][cells])[0, 1])
                effects = {}
                for name, values in (("raw", raw_scores[module]), ("filled", scores[module])):
                    per_donor = np.bincount(donor_index[cells], weights=values[cells], minlength=len(donors))
                    per_donor /= np.bincount(donor_index[cells], minlength=len(donors))
                    means = {c: (weights * m) @ per_donor / (weights * m).sum(axis=1) for c, m in (("SLE", sle), ("healthy", ~sle))}
                    pooled_sd = np.sqrt((np.var(per_donor[sle], ddof=1) + np.var(per_donor[~sle], ddof=1)) / 2)
                    effects[name] = means["SLE"] - means["healthy"]
                    effects[f"{name}_d"] = float(effects[name][0] / pooled_sd) if pooled_sd > 0 else float("nan")
                change = effects["filled"] - effects["raw"]
                q5_rows.append({**label, "module": module, "cells": cells_name, "per_cell_pearson_raw_vs_filled": r,
                                "effect_raw": float(effects["raw"][0]), "effect_raw_ci": interval(effects["raw"][1:]),
                                "effect_filled": float(effects["filled"][0]), "effect_filled_ci": interval(effects["filled"][1:]),
                                "effect_change": float(change[0]), "effect_change_ci": interval(change[1:]),
                                "cohen_d_raw": effects["raw_d"], "cohen_d_filled": effects["filled_d"]})

        if mode in ("global", "donor"):
            for group, cells in group_cells.items():
                filled_bulk = pseudobulk(matrix, donor_index, len(donors), cells)
                expressed = raw_bulk[group].max(axis=0) > 0
                effects = {}
                for name, bulk in (("raw", raw_bulk[group][:, expressed]), ("filled", filled_bulk[:, expressed])):
                    means = [(weights * m) @ bulk / (weights * m).sum(axis=1)[:, None] for m in (sle, ~sle)]
                    effects[name] = means[0] - means[1]
                x, y = effects["raw"], effects["filled"]
                xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
                r = (xc * yc).sum(axis=1) / np.sqrt((xc**2).sum(axis=1) * (yc**2).sum(axis=1))
                slope = (xc * yc).sum(axis=1) / (xc**2).sum(axis=1)
                sign_changes = ((np.sign(x) != np.sign(y)) & (x != 0)).sum(axis=1).astype(float)
                p_raw = welch_p(raw_bulk[group][:, expressed], sle[None, :])[0]
                p_filled = welch_p(filled_bulk[:, expressed], sle[None, :])[0]
                q_raw = stats.false_discovery_control(p_raw)
                q_filled = stats.false_discovery_control(p_filled)
                sig_raw, sig_filled = q_raw <= ALPHA, q_filled <= ALPHA
                q4_effects.append({**label, "cells": group, "genes": int(expressed.sum()),
                                   "pearson_effects": float(r[0]), "pearson_effects_ci": interval(r[1:]),
                                   "slope_filled_on_raw": float(slope[0]), "slope_ci": interval(slope[1:]),
                                   "sign_changes": int(sign_changes[0]), "sign_changes_ci": interval(sign_changes[1:]),
                                   "significant_raw": int(sig_raw.sum()), "significant_filled": int(sig_filled.sum()),
                                   "significant_both": int((sig_raw & sig_filled).sum()),
                                   "significance_changed": int((sig_raw != sig_filled).sum()),
                                   "significant_changed_sign": int((sig_raw & sig_filled & (np.sign(x[0]) != np.sign(y[0]))).sum())})
                if group in NULL_GROUPS:
                    values = cell_log(matrix[cells])
                    for kind, m in membership.items():
                        counts = {"wilcoxon": bh_count(wilcoxon_p(values, m[:, donor_index[cells]].astype(float))),
                                  "pseudobulk": bh_count(welch_p(filled_bulk, m))}
                        for test, count in counts.items():
                            reference = raw_null[group][kind][test]
                            row = {**label, "cells": group, "test": test, "labels": kind,
                                   "unfilled_mean": float(reference.mean()), "filled_mean": float(count.mean())}
                            if kind == "null":
                                difference = count - reference
                                boot = np.random.default_rng(args.seed).integers(0, len(difference), (args.draws, len(difference)))
                                row.update({
                                    "unfilled_median": float(np.median(reference)), "filled_median": float(np.median(count)),
                                    "unfilled_p95": float(np.quantile(reference, 0.95)), "filled_p95": float(np.quantile(count, 0.95)),
                                    "unfilled_any": float((reference > 0).mean()), "filled_any": float((count > 0).mean()),
                                    "mean_difference": float(difference.mean()), "mean_difference_ci": interval(difference[boot].mean(axis=1)),
                                    "permutations": int(len(count)),
                                })
                            q4_null.append(row)
        print(f"done {method} {mode} {level}", flush=True)

    tables = {"q2_treg_positive": q2_positive, "q2_fill_share_by_lineage": q2_share, "q2_fill_concentration": q2_conc,
              "q4_effects": q4_effects, "q4_null_de": q4_null, "q5_modules": q5_rows, "q6_donor_agreement": q6_rows}
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    summary["null_permutations"] = int(args.permutations)
    summary["raw_de_true_labels"] = {group: {test: int(values["observed"][test][0]) for test in ("wilcoxon", "pseudobulk")}
                                     for group, values in raw_null.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
