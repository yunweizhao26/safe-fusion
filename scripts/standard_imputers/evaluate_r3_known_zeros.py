#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

import evaluate_knockdown_zero_analyses as kd
from compute_matched_baseline_f1_curves import tie_broken_order
from disease_control_common import FRACTIONS as SEX_FRACTIONS
from disease_control_common import OUTPUT as DISEASE_CONTROL
from disease_control_common import fill_contract, interval, load_heldout, stratified_draws, unit_directory
from disease_control_sex_zeros import DEPLOYMENT_TEACHERS, SEX_GENES, selector_scores, summarize, zero_table
from evaluate_perturbation_zeros import FRACTIONS, Screen, eligible_targets, likely_dropout_labels, load_screen
from masked_f1_units import COUNT_SCALE

R3_ROOT = REPOSITORY / "artifacts" / "paper_evidence" / "review_round3" / "comparators"
NEW_METHODS = {"DCA": "dca", "DCA (ZINB)": "dca_zinb", "scImpute": "scimpute", "scRecover": "screcover", "EnImpute": "enimpute", "scVI (ZINB)": "scvi_zinb"}
PROBABILITY_METHODS = ("DCA (ZINB)", "scImpute", "scRecover", "scVI (ZINB)")
KNOCKDOWN_SCREENS = ("adamson_crispri", "papalexi_eccite", "norman_crispra")
SEX_UNITS = {"pancreas": ("pancreas_0", "pancreas_1", "pancreas_2"), "colon": ("colon",)}
REFERENCE = "safe_fusion"

def ranked_method(name: str, rank_score: np.ndarray, value: np.ndarray, screen: Screen, gene_columns: np.ndarray, seed: int) -> kd.Method:

    recorded, test = screen.recorded, screen.test
    library = recorded.sum(axis=1, dtype=np.float64)
    zero_rows, zero_cols = np.where((recorded == 0) & test[:, None])
    candidate = np.maximum(value[zero_rows, zero_cols], 0.0)
    rank = np.empty(len(candidate), dtype=np.int64)
    rank[tie_broken_order(rank_score[zero_rows, zero_cols].astype(np.float32), seed)] = np.arange(len(candidate))
    first = np.full(len(candidate), np.inf, dtype=np.float32)
    for pct in reversed(FRACTIONS):
        first[rank < max(1, int(round(pct / 100 * len(candidate))))] = pct
    first[candidate <= 0] = np.inf
    order = np.full(recorded.shape, np.inf, dtype=np.float32)
    order[zero_rows, zero_cols] = first
    local = {int(g): j for j, g in enumerate(gene_columns)}

    def source(pct: int):
        chosen = first <= pct
        sums = library + np.bincount(zero_rows[chosen], weights=candidate[chosen], minlength=len(library))
        columns = recorded[:, gene_columns].astype(np.float64)
        hit = chosen & np.isin(zero_cols, gene_columns)
        for r, c, v in zip(zero_rows[hit], zero_cols[hit], candidate[hit]):
            columns[r, local[int(c)]] = v
        return sums, columns

    score = np.full(recorded.shape, np.nan, dtype=np.float32)
    score[zero_rows, zero_cols] = rank_score[zero_rows, zero_cols]
    return kd.Method(name, order, score, value, source)

def knockdown(args: argparse.Namespace) -> dict:
    target_rows, auroc_rows, effect_rows, missing = [], [], [], []
    for dataset in KNOCKDOWN_SCREENS:
        inputs = kd.screen_inputs(dataset, args)
        screen = load_screen(dataset, inputs.deploy, inputs.splits, inputs.prepared)
        targets = eligible_targets(screen, args.min_detection)
        library = screen.recorded.sum(axis=1, dtype=np.float64)
        log_library = np.log(library)
        gene_columns = np.array([item.g for item in targets])
        methods = kd.build_methods(screen, gene_columns, inputs, args.seed)
        for name, directory in NEW_METHODS.items():
            contract = args.fits_root / directory / f"deploy_{dataset}"
            if not (contract / "mean.npy").exists():
                missing.append(str(contract))
                continue
            value = kd.count_scale_contract(contract, screen, library)
            methods[name] = ranked_method(name, value, value, screen, gene_columns, args.seed)
            if (contract / "dropout_probability.npy").exists():
                probability = np.load(contract / "dropout_probability.npy").astype(np.float32)
                methods[f"{name} P(dropout)"] = ranked_method(f"{name} P(dropout)", probability, value, screen, gene_columns, args.seed)
        test, control = screen.test, screen.control

        def effect(sums: np.ndarray, column: np.ndarray, gene: str) -> float:
            cp10k = column / sums * 1e4
            return kd.mean_log2_ratio(cp10k[test & (screen.target == gene)], cp10k[test & control])

        filled = {name: {pct: method.fill_source(pct) for pct in kd.EFFECT_FRACTIONS} for name, method in methods.items()}
        recorded_columns = screen.recorded[:, gene_columns].astype(np.float64)
        for j, item in enumerate(targets):
            labels = likely_dropout_labels(screen, item)
            cells = item.cells
            recorded_effect = effect(library, recorded_columns[:, j], item.gene)
            target_rows.append({"dataset": dataset, "direction": screen.direction, "gene": item.gene,
                                "heldout_log2fc_recorded": recorded_effect})
            for name, method in methods.items():
                for score_type, score in kd.target_scores(method, item, library).items():
                    if score_type == "depth_normalized":
                        continue
                    strata, _ = kd.stratified_auroc(labels, score, log_library[cells], args.depth_strata)
                    auroc_rows.append({"dataset": dataset, "gene": item.gene, "method": name, "score_type": score_type,
                                       "none": kd.auroc(labels, score), "depth_strata": strata})
                for pct in kd.EFFECT_FRACTIONS:
                    sums, columns = filled[name][pct]
                    value = effect(sums, columns[:, j], item.gene)
                    effect_rows.append({"dataset": dataset, "gene": item.gene, "method": name, "fill_pct": pct,
                                        "log2fc_recorded": recorded_effect, "shift": value - recorded_effect})
        print(json.dumps({"dataset": dataset, "targets": len(targets), "methods": list(methods)}), flush=True)

    targets, aurocs, effects = pd.DataFrame(target_rows), pd.DataFrame(auroc_rows), pd.DataFrame(effect_rows)
    summary_rows = []
    for dataset in KNOCKDOWN_SCREENS:
        genes = targets.loc[targets.dataset == dataset, "gene"].to_numpy()
        draws = np.random.default_rng([args.seed, kd.SCOPES.index(dataset)]).integers(0, len(genes), size=(args.draws, len(genes)))
        frame = aurocs[aurocs.dataset == dataset]
        reference = {key: part.set_index("gene").loc[genes] for key, part in frame[frame.method == REFERENCE].groupby("score_type")}
        for (name, score_type), part in frame.groupby(["method", "score_type"], sort=False):
            part = part.set_index("gene").loc[genes]
            for adjustment in ("none", "depth_strata"):
                values = part[adjustment].to_numpy()
                row = {"dataset": dataset, "method": name, "score_type": score_type, "adjustment": adjustment,
                       "n_targets": len(genes), "auroc": kd.interval(values, draws)}
                ref = reference[score_type][adjustment].to_numpy()
                row["safe_fusion_minus_method"] = kd.interval(ref - values, draws)
                summary_rows.append(row)
        for (name, pct), part in effects[effects.dataset == dataset].groupby(["method", "fill_pct"], sort=False):
            part = part.set_index("gene").loc[genes]
            summary_rows.append({"dataset": dataset, "method": name, "score_type": "log2fc_shift", "adjustment": f"{pct}pct",
                                 "n_targets": len(genes), "auroc": kd.interval(part["shift"].to_numpy(), draws)})
    output = args.output_dir / "knockdown"
    output.mkdir(parents=True, exist_ok=True)
    aurocs.to_csv(output / "auroc.csv", index=False)
    effects.to_csv(output / "effects.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    for column in ("auroc", "safe_fusion_minus_method"):
        for index, part in enumerate(("estimate", "lower", "upper")):
            summary[f"{column}_{part}"] = summary[column].map(lambda value, i=index: value[i] if isinstance(value, list) else np.nan)
    summary.drop(columns=["auroc", "safe_fusion_minus_method"]).rename(columns={
        "auroc_estimate": "value", "auroc_lower": "value_lower", "auroc_upper": "value_upper"}).to_csv(output / "summary.csv", index=False, float_format="%.4f")
    return {"missing": missing}

def sex_zeros(args: argparse.Namespace) -> dict:
    missing = []
    for tissue, keys in SEX_UNITS.items():
        data = load_heldout(tissue)
        symbols = list(data.symbols)
        gene_index = {gene: symbols.index(gene) for gene in SEX_GENES}
        donor_sex = pd.read_csv(DISEASE_CONTROL / "sex_zeros" / tissue / "donor_sex.csv", dtype={"donor": str})
        data.obs["sex"] = data.obs["donor"].map(donor_sex.set_index("donor")["sex"])
        cells = data.cells
        donors = sorted(cells["donor"].unique())
        draws = stratified_draws(cells.groupby("donor")["sex"].first().loc[donors].to_numpy(), args.draws, args.seed)
        table = zero_table(cells, data.counts, gene_index, None)
        full_rows = np.flatnonzero(data.heldout)[table["row"].to_numpy()]
        cols = table["col"].to_numpy()
        methods = ["Safe Fusion", *DEPLOYMENT_TEACHERS]
        for method in methods:
            score = np.full(len(table), np.nan)
            for fraction in SEX_FRACTIONS:
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
        library = data.recorded.sum(axis=1, dtype=np.float64)
        for name, directory in NEW_METHODS.items():
            for ranking in ("value", "dropout"):
                label = name if ranking == "value" else f"{name} P(dropout)"
                fill_name = directory if ranking == "value" else f"{directory}_dropout"
                contracts = {key: args.fits_root / directory / f"deploy_{key}" for key in keys}
                fills = {key: args.fills_root / key for key in keys}
                if ranking == "dropout" and name not in PROBABILITY_METHODS:
                    continue
                needed = [contract / ("mean.npy" if ranking == "value" else "dropout_probability.npy") for contract in contracts.values()]
                needed += [fills[key] / f"{fill_name}_topk_{f'{fraction:g}'.replace('.', 'p')}" / "mean.npy" for key in keys for fraction in SEX_FRACTIONS]
                absent = [str(path) for path in needed if not path.exists()]
                if absent:
                    missing += absent
                    continue
                score = np.full(len(table), np.nan)
                for fraction in SEX_FRACTIONS:
                    filled = np.zeros(len(table), dtype=bool)
                    for key, test in data.test_masks.items():
                        mine = test[full_rows]
                        values = np.load(fills[key] / f"{fill_name}_topk_{f'{fraction:g}'.replace('.', 'p')}" / "mean.npy", mmap_mode="r")
                        filled[mine] = values[full_rows[mine], cols[mine]] != 0
                    table[f"filled:{label}:{fraction}"] = filled
                for key, test in data.test_masks.items():
                    mine = test[full_rows]
                    metadata = json.loads((contracts[key] / "metadata.json").read_text())
                    if ranking == "value":
                        mean = np.load(contracts[key] / "mean.npy", mmap_mode="r")
                        raw = np.asarray(mean[full_rows[mine], cols[mine]], dtype=np.float64)
                        score[mine] = np.maximum(COUNT_SCALE[metadata["scale"]](raw, library[full_rows[mine]]), 0.0)
                    else:
                        probability = np.load(contracts[key] / "dropout_probability.npy", mmap_mode="r")
                        score[mine] = np.asarray(probability[full_rows[mine], cols[mine]], dtype=np.float64)
                table[f"score:{label}"] = score
                methods.append(label)
        fill_rows, auroc_rows = summarize(table, methods, ["recorded zero, expressing sex"], draws, donors, "recorded counts")
        auroc = pd.DataFrame(auroc_rows)
        differences = paired_auroc_differences(table, methods, draws, donors)
        output = args.output_dir / "sex_zeros" / tissue
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(fill_rows).to_csv(output / "fill_rates.csv", index=False)
        auroc.merge(differences, on=["gene", "method"], how="left").to_csv(output / "auroc.csv", index=False, float_format="%.4f")
        print(auroc[["gene", "method", "auroc", "ci_low", "ci_high"]].to_string(index=False), flush=True)
    return {"missing": missing}

def paired_auroc_differences(table: pd.DataFrame, methods: list[str], draws: np.ndarray, donors: list[str]) -> pd.DataFrame:

    donor_code = pd.Categorical(table["donor"], categories=donors).codes
    weights = np.stack([np.bincount(draw, minlength=len(donors))[donor_code] for draw in draws]).astype(float)
    rows = []
    for gene, index in table.groupby("gene").groups.items():
        index = np.asarray(index)
        kind = table["kind"].to_numpy()[index]
        keep = (kind == "recorded zero, expressing sex") | (kind == "absent, other sex")
        label = (kind[keep] == "recorded zero, expressing sex").astype(int)
        w = weights[:, index][:, keep]
        reference = table["score:Safe Fusion"].to_numpy()[index][keep]
        for method in methods:
            if method == "Safe Fusion":
                continue
            score = table[f"score:{method}"].to_numpy()[index][keep]
            point = roc_auc_score(label, reference) - roc_auc_score(label, score)
            boot = []
            for row in w:
                present = row > 0
                if len(np.unique(label[present])) < 2:
                    boot.append(np.nan)
                    continue
                boot.append(roc_auc_score(label[present], reference[present], sample_weight=row[present])
                            - roc_auc_score(label[present], score[present], sample_weight=row[present]))
            low, high = interval(np.asarray(boot))
            rows.append({"gene": gene, "method": method, "safe_fusion_minus_method": point,
                         "difference_ci_low": low, "difference_ci_high": high})
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits-root", type=Path, default=R3_ROOT / "fits")
    parser.add_argument("--fills-root", type=Path, default=R3_ROOT / "deployment_fills")
    parser.add_argument("--output-dir", type=Path, default=R3_ROOT / "evaluation" / "known_zeros")
    parser.add_argument("--deploy-root", default="artifacts/paper_evidence/downstream_deployment")
    parser.add_argument("--review-root", default="artifacts/paper_evidence/review_round2/knockdown")
    parser.add_argument("--norman-benchmark", default="artifacts/paper_evidence/review_round2/leakage_free/norman_crispra")
    parser.add_argument("--norman-root", default="artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt")
    parser.add_argument("--min-detection", type=float, default=0.2)
    parser.add_argument("--depth-strata", type=int, default=5)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--parts", nargs="+", choices=["knockdown", "sex"], default=["knockdown", "sex"])
    args = parser.parse_args()

    report = {}
    if "knockdown" in args.parts:
        report["knockdown"] = knockdown(args)
    if "sex" in args.parts:
        report["sex_zeros"] = sex_zeros(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"report_{'_'.join(args.parts)}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
