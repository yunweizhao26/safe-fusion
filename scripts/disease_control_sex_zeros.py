#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from compute_matched_baseline_f1_curves import tie_broken_order
from disease_control_common import (
    DRAWS,
    FRACTIONS,
    METHODS,
    OUTPUT,
    REPOSITORY,
    SEED,
    TISSUES,
    fill_contract,
    interval,
    load_heldout,
    output_dir,
    stratified_draws,
    unit_directory,
)
from masked_f1_units import (
    COLON_METHODS,
    EVIDENCE,
    MAIN_COMPARATORS,
    SUPPLEMENTARY_COMPARATORS,
    TEACHER_COMPARATORS,
    build_units,
    count_scale_values,
    fraction_name,
    load_unit,
)

SEX_GENES = {"XIST": "female", "RPS4Y1": "male"}
DEPLOYMENT_TEACHERS = {"SVD": "svd_impute", "Weighted kNN": "graph_smooth", "MAGIC": "magic_inductive", "scVI": "scvi_inductive"}


def selector_scores(setting: str, key: str) -> pd.DataFrame:
    frame = pd.read_parquet(OUTPUT / "selector_scores" / setting / key / "selected_gene_scores.parquet")
    return frame[frame["split"] == "test"].set_index(["cell_index", "gene_index"])["selector_score"]


def refit_check(setting: str, key: str, production_dir, rows: np.ndarray, cols: np.ndarray) -> list[dict]:
    result = []
    for fraction in FRACTIONS:
        name = f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}"
        refit = np.load(OUTPUT / "selector_scores" / setting / key / name / "mean.npy", mmap_mode="r")[rows, cols] != 0
        production = np.load(production_dir / name / "mean.npy", mmap_mode="r")[rows, cols] != 0
        result.append({
            "setting": setting, "unit": key, "fraction": fraction, "n_zeros": int(len(rows)),
            "filled_refit": int(refit.sum()), "filled_production": int(production.sum()),
            "filled_by_one_only": int((refit != production).sum()),
        })
    return result


def zero_table(cells: pd.DataFrame, counts: np.ndarray, gene_index: dict, masked: np.ndarray | None) -> pd.DataFrame:
    frames = []
    for gene, expressing in SEX_GENES.items():
        col = gene_index[gene]
        zero = np.flatnonzero(counts[:, col] == 0)
        is_masked = masked[zero, col] if masked is not None else np.zeros(len(zero), dtype=bool)
        expressing_sex = cells["sex"].to_numpy()[zero] == expressing
        kind = np.where(expressing_sex, np.where(is_masked, "hidden count, expressing sex", "recorded zero, expressing sex"),
                        np.where(is_masked, "hidden count, other sex", "absent, other sex"))
        frames.append(pd.DataFrame({"gene": gene, "row": zero, "col": col, "kind": kind,
                                    "donor": cells["donor"].to_numpy()[zero], "sex": cells["sex"].to_numpy()[zero],
                                    "library_size": counts[zero].sum(axis=1)}))
    return pd.concat(frames, ignore_index=True)


def zero_counts(table: pd.DataFrame, setting: str) -> list[dict]:
    grouped = table.groupby(["gene", "kind"])
    frame = pd.DataFrame({"n_zeros": grouped.size(), "n_donors": grouped["donor"].nunique(),
                          "median_library_size": grouped["library_size"].median()}).reset_index()
    return frame.assign(setting=setting).to_dict("records")


def summarize(table: pd.DataFrame, methods: list[str], positives: list[str], draws: np.ndarray, donors: list[str], setting: str) -> tuple[list, list]:
    donor_code = pd.Categorical(table["donor"], categories=donors).codes
    weights = np.stack([np.bincount(draw, minlength=len(donors))[donor_code] for draw in draws]).astype(float)
    fill_rows, auroc_rows = [], []
    kinds = sorted(table["kind"].unique())
    for gene, frame_index in table.groupby("gene").groups.items():
        index = np.asarray(frame_index)
        kind = table["kind"].to_numpy()[index]
        w = weights[:, index]
        negative = kind == "absent, other sex"
        for method in methods:
            for fraction in FRACTIONS:
                filled = table[f"filled:{method}:{fraction}"].to_numpy()[index].astype(float)
                rates = {}
                for level in kinds:
                    mask = kind == level
                    point = float(filled[mask].mean()) if mask.any() else float("nan")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        boot = (w[:, mask] @ filled[mask]) / w[:, mask].sum(axis=1)
                    rates[level] = boot
                    low, high = interval(boot)
                    fill_rows.append({"setting": setting, "gene": gene, "method": method, "fraction": fraction, "kind": level,
                                      "n_zeros": int(mask.sum()), "fill_rate": point, "ci_low": low, "ci_high": high})
                for positive in positives:
                    mask = kind == positive
                    if not mask.any() or not negative.any():
                        continue
                    difference = rates[positive] - rates["absent, other sex"]
                    low, high = interval(difference)
                    fill_rows.append({"setting": setting, "gene": gene, "method": method, "fraction": fraction,
                                      "kind": f"difference: {positive} minus absent", "n_zeros": int(mask.sum() + negative.sum()),
                                      "fill_rate": float(filled[mask].mean() - filled[negative].mean()), "ci_low": low, "ci_high": high})
            score = table[f"score:{method}"].to_numpy()[index]
            for positive in positives:
                mask = (kind == positive) | negative
                if not (kind == positive).any() or not negative.any():
                    continue
                label = (kind[mask] == positive).astype(int)
                point = roc_auc_score(label, score[mask])
                boot = []
                for row in w[:, mask]:
                    keep = row > 0
                    boot.append(roc_auc_score(label[keep], score[mask][keep], sample_weight=row[keep])
                                if len(np.unique(label[keep])) == 2 else float("nan"))
                low, high = interval(np.asarray(boot))
                auroc_rows.append({"setting": setting, "gene": gene, "method": method, "positive": positive,
                                   "negative": "absent, other sex", "n_positive": int((kind == positive).sum()),
                                   "n_negative": int(negative.sum()), "auroc": float(point), "ci_low": low, "ci_high": high})
    return fill_rows, auroc_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    spec = TISSUES[args.tissue]
    output = output_dir("sex_zeros", args.tissue)

    data = load_heldout(args.tissue)
    symbols = list(data.symbols)
    gene_index = {gene: symbols.index(gene) for gene in SEX_GENES}
    every = data.obs
    library = data.recorded.sum(axis=1, dtype=np.float64)
    sex_rows = []
    for donor, members in every.groupby("donor").groups.items():
        rows = np.asarray(members)
        total = library[rows].sum()
        sex_rows.append({"donor": donor, "metadata_sex": every.loc[rows[0], "sex"], "heldout": bool(every.loc[rows, "heldout"].any()),
                         "n_cells": int(len(rows)),
                         **{f"{gene}_cpm": float(1e6 * data.recorded[rows, col].sum() / total) for gene, col in gene_index.items()},
                         **{f"{gene}_detected_share": float(np.mean(data.recorded[rows, col] > 0)) for gene, col in gene_index.items()}})
    sex_table = pd.DataFrame(sex_rows)
    sex_table["sex"] = np.where(sex_table["XIST_cpm"] > sex_table["RPS4Y1_cpm"], "female", "male")
    sex_table["metadata_agrees"] = sex_table["sex"] == sex_table["metadata_sex"]
    sex_table.to_csv(output / "donor_sex.csv", index=False)
    data.obs["sex"] = data.obs["donor"].map(sex_table.set_index("donor")["sex"])

    cells = data.cells
    donors = sorted(cells["donor"].unique())
    donor_sex = cells.groupby("donor")["sex"].first().loc[donors].to_numpy()
    draws = stratified_draws(donor_sex, args.draws, args.seed)
    heldout_rows = np.flatnonzero(data.heldout)
    fill_rows, auroc_rows, check_rows, count_rows = [], [], [], []

    table = zero_table(cells, data.counts, gene_index, None)
    full_rows = heldout_rows[table["row"].to_numpy()]
    cols = table["col"].to_numpy()
    for method in METHODS:
        score = np.full(len(table), np.nan)
        for fraction in FRACTIONS:
            filled = np.zeros(len(table), dtype=bool)
            for key, test in data.test_masks.items():
                mine = test[full_rows]
                values = np.load(fill_contract(key, method, fraction) / "mean.npy", mmap_mode="r")
                filled[mine] = values[full_rows[mine], cols[mine]] != 0
            table[f"filled:{method}:{fraction}"] = filled
        for key, test in data.test_masks.items():
            mine = test[full_rows]
            if method == "Safe Fusion":
                scores = selector_scores("deployment", key)
                score[mine] = scores.loc[list(zip(full_rows[mine], cols[mine]))].to_numpy()
            else:
                values = np.load(unit_directory(key) / "methods" / DEPLOYMENT_TEACHERS[method] / "mean.npy", mmap_mode="r")
                score[mine] = np.maximum(values[full_rows[mine], cols[mine]], 0.0)
        table[f"score:{method}"] = score
    for key, test in data.test_masks.items():
        rows, cols_all = np.where((data.recorded == 0) & test[:, None])
        check_rows += refit_check("deployment", key, unit_directory(key) / "selector", rows, cols_all)
    count_rows += zero_counts(table, "recorded counts")
    rows_out = summarize(table, list(METHODS), ["recorded zero, expressing sex"], draws, donors, "recorded counts")
    fill_rows += rows_out[0]
    auroc_rows += rows_out[1]

    comparators = [*MAIN_COMPARATORS, *SUPPLEMENTARY_COMPARATORS, *TEACHER_COMPARATORS]
    units = [unit for unit in build_units(EVIDENCE, COLON_METHODS) if unit.key in spec["units"]]
    frames = []
    for unit in units:
        unit_data = load_unit(unit)
        if unit_data.cell_ids != data.obs["cell_id"].tolist():
            raise ValueError("masked benchmark cell order differs from the recorded matrix")
        test_cells = np.flatnonzero(unit_data.split == "test")
        local_rows, zero_cols = np.where(unit_data.counts[test_cells] == 0)
        all_rows = test_cells[local_rows]
        n = len(all_rows)
        unit_cells = data.obs.iloc[test_cells].reset_index(drop=True)
        unit_table = zero_table(unit_cells, unit_data.counts[test_cells], gene_index, unit_data.masked[test_cells])
        unit_rows = test_cells[unit_table["row"].to_numpy()]
        unit_cols = unit_table["col"].to_numpy()
        lookup = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([all_rows, zero_cols]))
        where = lookup.loc[list(zip(unit_rows, unit_cols))].to_numpy()
        for name in comparators:
            scores, _ = count_scale_values(unit.contracts[name], unit_data, all_rows, zero_cols)
            order = tie_broken_order(scores, unit.tie_seed)
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(n)
            unit_table[f"score:{name}"] = scores[where]
            for fraction in FRACTIONS:
                unit_table[f"filled:{name}:{fraction}"] = rank[where] < max(1, int(round(fraction * n)))
        selector = selector_scores("masked", unit.key)
        unit_table["score:Safe Fusion"] = selector.loc[list(zip(unit_rows, unit_cols))].to_numpy()
        for fraction in FRACTIONS:
            contract = unit.selector_dir / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}"
            unit_table[f"filled:Safe Fusion:{fraction}"] = np.load(contract / "mean.npy", mmap_mode="r")[unit_rows, unit_cols] != 0
        check_rows += refit_check("masked", unit.key, unit.selector_dir, all_rows, zero_cols)
        frames.append(unit_table)
    table = pd.concat(frames, ignore_index=True)
    count_rows += zero_counts(table, "masked benchmark")
    rows_out = summarize(table, ["Safe Fusion", *comparators], ["hidden count, expressing sex", "recorded zero, expressing sex"], draws, donors, "masked benchmark")
    fill_rows += rows_out[0]
    auroc_rows += rows_out[1]

    pd.DataFrame(count_rows).to_csv(output / "zero_counts.csv", index=False)
    pd.DataFrame(fill_rows).to_csv(output / "fill_rates.csv", index=False)
    pd.DataFrame(auroc_rows).to_csv(output / "auroc.csv", index=False)
    pd.DataFrame(check_rows).to_csv(output / "selector_refit_check.csv", index=False)
    (output / "design.json").write_text(json.dumps({
        "genes": SEX_GENES,
        "donor_sex": "female when the donor's pooled recorded counts have more XIST than RPS4Y1 per million, male otherwise",
        "metadata_sex_source": str(spec["source"].relative_to(REPOSITORY)),
        "bootstrap": f"{args.draws} draws of held-out donors within each sex", "seed": args.seed,
    }, indent=2) + "\n")
    print(pd.DataFrame(count_rows).to_string(index=False))
    print(pd.DataFrame(auroc_rows).to_string(index=False))
    print(pd.DataFrame(check_rows).to_string(index=False))


if __name__ == "__main__":
    main()
