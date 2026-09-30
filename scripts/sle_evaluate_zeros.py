#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sle_common import (
    CASE, FOLDS, METHODS, NOT_FILLED, REPORTED, bootstrap_draws, config, interval, load_fills, load_unit,
    pair_auroc, weighted_auroc,
)

MIN_DETECTION = 0.20
DEPTH_BINS = 5

def tests(set_name: str, obs: pd.DataFrame, genes: set[str], settings: dict) -> list[dict]:

    items = []
    sexes = set(obs.loc[obs["q1_included"], "sex"])
    if {"male", "female"} <= sexes:
        for gene in settings["sex_x"]:
            items.append({"family": "sex", "gene": gene, "positive": "female", "negative": "male", "column": "sex"})
        for gene in settings["sex_y"]:
            items.append({"family": "sex", "gene": gene, "positive": "male", "negative": "female", "column": "sex"})
    elif sexes == {"female"}:
        for gene in settings["sex_x"]:
            items.append({"family": "sex_one_sex", "gene": gene, "positive": "female", "negative": "none", "column": "sex"})
        for gene in settings["sex_y"]:
            items.append({"family": "sex_one_sex", "gene": gene, "positive": "none", "negative": "female", "column": "sex"})
    for lineage in settings["treg_exclusive_lineages"]:
        for gene in settings["lineage"][lineage]:
            items.append({"family": "lineage", "gene": gene, "positive": lineage, "negative": "Treg", "column": "lineage"})
    return [item for item in items if item["gene"] in genes]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    output = args.output_dir or CASE / "results" / f"q1_zeros_{args.set_name}"
    settings = config()
    report = json.loads((CASE / args.set_name / "prepare_report.json").read_text())
    sex_table = pd.DataFrame(report["sex_by_expression"]).set_index("donor")
    discordant = sex_table.index[sex_table["metadata_sex"] != sex_table["expression_sex"]].tolist()
    strata_column = "sex" if report["design"]["contrast"] == "sex" else "condition"

    zero_records = []
    all_donors = {}
    for fold in range(FOLDS):
        unit = load_unit(args.set_name, fold, "deploy")
        obs = unit.obs.copy()
        obs["q1_included"] = ~obs["donor"].isin(discordant)
        all_donors.update(obs.loc[obs["q1_included"]].drop_duplicates("donor").set_index("donor")[strata_column].to_dict())
        fills = {method: load_fills(unit, method) for method in METHODS}
        rows, cols = fills["Safe Fusion"]["rows"], fills["Safe Fusion"]["cols"]
        library = unit.counts.sum(axis=1)
        test_library = library[unit.test]
        edges = np.quantile(test_library, np.linspace(0, 1, DEPTH_BINS + 1)[1:-1])
        depth_bin = np.searchsorted(edges, library, side="right")
        for item in tests(args.set_name, obs, set(unit.genes), settings):
            g = unit.genes.index(item["gene"])
            in_gene = np.flatnonzero(cols == g)
            cell_rows = rows[in_gene]
            group = obs[item["column"]].astype(str).to_numpy()[cell_rows]
            keep = obs["q1_included"].to_numpy()[cell_rows] & np.isin(group, [item["positive"], item["negative"]])
            frame = pd.DataFrame({
                "fold": fold, "family": item["family"], "gene": item["gene"],
                "donor": obs["donor"].astype(str).to_numpy()[cell_rows[keep]],
                "likely_dropout": (group[keep] == item["positive"]).astype(np.int8),
                "depth_bin": depth_bin[cell_rows[keep]],
                "library": library[cell_rows[keep]],
            })
            for method, data in fills.items():
                first = np.where(data["value"][in_gene] > 0, data["first_global"][in_gene], NOT_FILLED)[keep]
                frame[f"first|{method}"] = first
                frame[f"priority|{method}"] = data["priority"][in_gene][keep]
            zero_records.append(frame)
    zeros = pd.concat(zero_records, ignore_index=True)

    prepared = load_unit(args.set_name, 0, "deploy")
    expressed = {}
    obs = prepared.obs.copy()
    included = ~obs["donor"].isin(discordant).to_numpy()
    for item in tests(args.set_name, obs.assign(q1_included=included), set(prepared.genes), settings):
        g = prepared.genes.index(item["gene"])
        cells = included & (obs[item["column"]].astype(str).to_numpy() == item["positive"])
        other = included & (obs[item["column"]].astype(str).to_numpy() == item["negative"])
        expressed[item["gene"]] = {
            "detection_in_expressing_cells": float((prepared.recorded[cells, g] > 0).mean()) if cells.any() else None,
            "detection_in_nonexpressing_cells": float((prepared.recorded[other, g] > 0).mean()) if other.any() else None,
        }

    donors = np.asarray(sorted(all_donors))
    strata = np.asarray([all_donors[donor] for donor in donors])
    weights = np.vstack([np.ones(len(donors)), bootstrap_draws(donors, strata, args.draws)])
    rows_out, auroc_draws = [], {}
    for (family, gene), frame in zeros.groupby(["family", "gene"], sort=False):
        donor_index = pd.Index(donors).get_indexer(frame["donor"])
        positive = frame["likely_dropout"].to_numpy() == 1
        n_pos = np.bincount(donor_index[positive], minlength=len(donors)).astype(float)
        n_neg = np.bincount(donor_index[~positive], minlength=len(donors)).astype(float)
        for method in METHODS:
            first = frame[f"first|{method}"].to_numpy()
            result = {"family": family, "gene": gene, "method": method,
                      "n_likely_dropout_zeros": int(positive.sum()), "n_biological_zeros": int((~positive).sum()),
                      **expressed.get(gene, {})}
            for level in REPORTED:
                filled = first <= level
                pos_fill = np.bincount(donor_index[positive & filled], minlength=len(donors))
                neg_fill = np.bincount(donor_index[~positive & filled], minlength=len(donors))
                rate_pos = (weights @ pos_fill) / (weights @ n_pos)
                rate_neg = (weights @ neg_fill) / (weights @ n_neg)
                result[f"fill_likely_dropout_{level}pct"] = rate_pos[0]
                result[f"fill_likely_dropout_{level}pct_ci"] = interval(rate_pos[1:])
                result[f"fill_biological_{level}pct"] = rate_neg[0]
                result[f"fill_biological_{level}pct_ci"] = interval(rate_neg[1:])
                result[f"fill_difference_{level}pct"] = rate_pos[0] - rate_neg[0]
                result[f"fill_difference_{level}pct_ci"] = interval((rate_pos - rate_neg)[1:])
            for kind, values in (("paper", np.where(first <= 10, 11 - first.astype(float), 0.0)),
                                 ("continuous", frame[f"priority|{method}"].to_numpy(float))):
                pos_lists = [values[positive & (donor_index == d)] for d in range(len(donors))]
                neg_lists = [values[~positive & (donor_index == d)] for d in range(len(donors))]
                auroc = weighted_auroc(pair_auroc(pos_lists, neg_lists), n_pos, n_neg, weights, weights)
                auroc_draws[(family, gene, method, kind)] = auroc
                result[f"auroc_{kind}"] = auroc[0]
                result[f"auroc_{kind}_ci"] = interval(auroc[1:])
            values = np.where(first <= 10, 11 - first.astype(float), 0.0)
            numerator = np.zeros(len(weights))
            denominator = np.zeros(len(weights))
            depth = frame["depth_bin"].to_numpy()
            for b in range(DEPTH_BINS):
                in_bin = depth == b
                pos_lists = [values[positive & in_bin & (donor_index == d)] for d in range(len(donors))]
                neg_lists = [values[~positive & in_bin & (donor_index == d)] for d in range(len(donors))]
                u = pair_auroc(pos_lists, neg_lists)
                pos_n = np.asarray([len(v) for v in pos_lists], float)
                neg_n = np.asarray([len(v) for v in neg_lists], float)
                numerator += np.einsum("ka,ab,kb->k", weights, u, weights)
                denominator += (weights @ pos_n) * (weights @ neg_n)
            auroc = np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator > 0)
            auroc_draws[(family, gene, method, "paper_depth_stratified")] = auroc
            result["auroc_paper_depth_stratified"] = auroc[0]
            result["auroc_paper_depth_stratified_ci"] = interval(auroc[1:])
            result["median_library_likely_dropout"] = float(np.median(frame.loc[positive, "library"])) if positive.any() else np.nan
            result["median_library_biological"] = float(np.median(frame.loc[~positive, "library"])) if (~positive).any() else np.nan
            rows_out.append(result)
    per_gene = pd.DataFrame(rows_out)

    summary = {
        "set": args.set_name, "donors": len(donors), "donors_by_stratum": pd.Series(strata).value_counts().to_dict(),
        "excluded_discordant_sex_donors": discordant, "min_detection_for_pooled_sex_auroc": MIN_DETECTION,
        "pooled": {},
    }
    for family in per_gene["family"].unique():
        genes = per_gene.loc[per_gene["family"] == family, "gene"].unique().tolist()
        if family == "sex_one_sex":
            continue
        if family == "sex":
            genes = [gene for gene in genes if expressed[gene]["detection_in_expressing_cells"] >= MIN_DETECTION]
        for kind in ("paper", "continuous", "paper_depth_stratified"):
            pooled = {method: np.nanmean([auroc_draws[(family, gene, method, kind)] for gene in genes], axis=0) for method in METHODS}
            block = {"genes": genes}
            for method, values in pooled.items():
                block[method] = {"auroc": float(values[0]), "ci": interval(values[1:])}
                if method != "Safe Fusion":
                    difference = pooled["Safe Fusion"] - values
                    block[method]["safe_fusion_minus_method"] = float(difference[0])
                    block[method]["safe_fusion_minus_method_ci"] = interval(difference[1:])
            summary["pooled"][f"{family}|{kind}"] = block
    output.mkdir(parents=True, exist_ok=True)
    per_gene.to_csv(output / "per_gene.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(per_gene[["family", "gene", "method", "n_likely_dropout_zeros", "n_biological_zeros", "fill_likely_dropout_5pct",
                    "fill_biological_5pct", "auroc_paper", "auroc_continuous", "auroc_paper_depth_stratified"]].to_string(index=False))
    print(json.dumps(summary, indent=2, default=float))

if __name__ == "__main__":
    main()
