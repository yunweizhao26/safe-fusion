#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from compute_matched_baseline_f1_curves import tie_broken_order
from evaluate_perturbation_zeros import (
    FRACTIONS, SCREENS, SPARSE_METHODS, SPLITS, Screen, Target, eligible_targets, first_fill_order,
    fraction_suffix, likely_dropout_labels, load_screen, mean_log2_ratio,
)
from masked_f1_units import COUNT_SCALE, stored_scale

TEACHER_VALUES = {
    "svd": "svd_impute",
    "weighted_knn": "graph_smooth",
    "magic": "magic_inductive",
    "scvi": "scvi_inductive",
    "knn_condition": "graph_smooth_condition",
    "scvi_condition": "scvi_inductive_condition",
}
SELECTORS = ("safe_fusion", "safe_fusion_condition")
STANDARD = {"alra": "alra", "magic_standard": "magic", "scvi_standard": "scvi", "saver": "saver"}
LABEL_METHODS = ("safe_fusion_condition", "knn_condition", "scvi_condition")
METHOD_ORDER = ("safe_fusion", "svd", "weighted_knn", "magic", "scvi", "alra", "magic_standard", "scvi_standard", "saver",
                "safe_fusion_condition", "knn_condition", "scvi_condition")
SCORE_TYPES = ("fill_order", "continuous", "depth_normalized")
ADJUSTMENTS = ("none", "depth_strata", "depth_matched")
EFFECT_FRACTIONS = (1, 5, 10)
KNOCKDOWN_COMPARISON = ("adamson_crispri", "papalexi_eccite")
SCOPES = (*SCREENS, "knockdown_pooled")


@dataclass
class ScreenInputs:
    prepared: Path
    splits: Path
    deploy: Path
    selector_scores: dict[str, Path]
    standard: dict[str, Path]


def screen_inputs(dataset: str, args: argparse.Namespace) -> ScreenInputs:
    if dataset == "norman_crispra":
        benchmark, root = Path(args.norman_benchmark), Path(args.norman_root)
        deploy = root / "deployment"
        return ScreenInputs(
            prepared=benchmark / "prepared.h5ad",
            splits=benchmark / "splits.parquet",
            deploy=deploy,
            selector_scores={"safe_fusion": deploy / "selector" / "selected_gene_scores.parquet",
                             "safe_fusion_condition": deploy / "selector_condition" / "selected_gene_scores.parquet"},
            standard={name: root / "standard_imputers" / directory for name, directory in STANDARD.items()},
        )
    review = Path(args.review_root)
    return ScreenInputs(
        prepared=Path(f"external_data/prepared/{dataset}.h5ad"),
        splits=SPLITS[dataset],
        deploy=Path(args.deploy_root) / dataset,
        selector_scores={name: review / "selector_scores" / dataset / name / "selected_gene_scores.parquet" for name in SELECTORS},
        standard={name: review / "standard_imputers" / directory / dataset for name, directory in STANDARD.items()},
    )


class Method:
    def __init__(self, name: str, order: np.ndarray, score: np.ndarray | None, value: np.ndarray | None, fill_source):
        self.name = name
        self.order = order
        self.score = score
        self.value = value
        self.fill_source = fill_source


def count_scale_contract(contract: Path, screen: Screen, library: np.ndarray) -> np.ndarray:
    metadata = json.loads((contract / "metadata.json").read_text())
    if metadata["cell_ids"] != screen.cell_ids.tolist():
        raise ValueError(f"cell order differs for {contract}")
    scale = stored_scale(metadata)
    mean = np.asarray(np.load(contract / "mean.npy"), dtype=np.float64)
    return COUNT_SCALE[scale](mean, library[:, None]).astype(np.float32)


def selector_score_matrix(path: Path, shape: tuple[int, int]) -> np.ndarray:
    frame = pd.read_parquet(path, columns=["split", "cell_index", "gene_index", "selector_score"])
    frame = frame[frame["split"] == "test"]
    score = np.full(shape, np.nan, dtype=np.float32)
    score[frame["cell_index"].to_numpy(), frame["gene_index"].to_numpy()] = frame["selector_score"].to_numpy()
    return score


def build_methods(screen: Screen, gene_columns: np.ndarray, inputs: ScreenInputs, seed: int) -> dict[str, Method]:
    deploy = inputs.deploy
    recorded, test = screen.recorded, screen.test
    library = recorded.sum(axis=1, dtype=np.float64)
    zero_rows, zero_cols = np.where((recorded == 0) & test[:, None])
    orders = first_fill_order(screen, deploy)
    methods: dict[str, Method] = {}

    def sparse_source(template: str):
        def source(pct: int):
            matrix = np.load(deploy / template.format(pct=pct, suffix=fraction_suffix(pct)) / "mean.npy", mmap_mode="r")
            return np.asarray(matrix, dtype=np.float64).sum(axis=1), np.asarray(matrix[:, gene_columns], dtype=np.float64)
        return source

    for name in SELECTORS:
        score = selector_score_matrix(inputs.selector_scores[name], recorded.shape)
        methods[name] = Method(name, orders[name], score, None, sparse_source(SPARSE_METHODS[name]))
    for name, contract in TEACHER_VALUES.items():
        value = count_scale_contract(deploy / "methods" / contract, screen, library)
        methods[name] = Method(name, orders[name], value, value, sparse_source(SPARSE_METHODS[name]))
    for name in STANDARD:
        value = count_scale_contract(inputs.standard[name], screen, library)
        candidate = np.maximum(value[zero_rows, zero_cols], 0.0)
        rank = np.empty(len(candidate), dtype=np.int64)
        rank[tie_broken_order(candidate, seed)] = np.arange(len(candidate))
        order = np.full(recorded.shape, np.inf, dtype=np.float32)
        first = np.full(len(candidate), np.inf, dtype=np.float32)
        for pct in reversed(FRACTIONS):
            first[rank < max(1, int(round(pct / 100 * len(candidate))))] = pct
        first[candidate <= 0] = np.inf
        order[zero_rows, zero_cols] = first

        def source(pct: int, first=first, candidate=candidate):
            chosen = first <= pct
            sums = library + np.bincount(zero_rows[chosen], weights=candidate[chosen], minlength=len(library))
            columns = recorded[:, gene_columns].astype(np.float64)
            local = {int(g): j for j, g in enumerate(gene_columns)}
            hit = chosen & np.isin(zero_cols, gene_columns)
            for r, c, v in zip(zero_rows[hit], zero_cols[hit], candidate[hit]):
                columns[r, local[int(c)]] = v
            return sums, columns

        methods[name] = Method(name, order, value, value, source)
    return methods


def auroc(labels: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(labels, score))


def stratified_auroc(labels: np.ndarray, score: np.ndarray, log_library: np.ndarray, strata: int) -> tuple[float, float]:
    edges = np.quantile(log_library, np.linspace(0, 1, strata + 1)[1:-1])
    stratum = np.searchsorted(edges, log_library, side="right")
    weighted, pairs = 0.0, 0.0
    for index in range(strata):
        members = stratum == index
        positives = labels[members].sum()
        negatives = members.sum() - positives
        if positives and negatives:
            weighted += positives * negatives * auroc(labels[members], score[members])
            pairs += positives * negatives
    total = labels.sum() * (len(labels) - labels.sum())
    return (weighted / pairs if pairs else float("nan")), float(pairs / total)


def matched_auroc(labels: np.ndarray, score: np.ndarray, log_library: np.ndarray) -> tuple[float, float]:
    dropout, biological = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    distance = np.abs(log_library[biological][:, None] - log_library[dropout][None, :])
    partner = dropout[np.argmin(distance, axis=1)]
    concordance = (score[partner] > score[biological]) + 0.5 * (score[partner] == score[biological])
    return float(concordance.mean()), float(np.median(np.min(distance, axis=1)))


def target_scores(method: Method, item: Target, library: np.ndarray) -> dict[str, np.ndarray]:
    cells, g = item.cells, item.g
    order = method.order[cells, g]
    scores = {"fill_order": np.where(np.isfinite(order), 11 - order, 0.0), "continuous": method.score[cells, g].astype(np.float64)}
    if np.isnan(scores["continuous"]).any():
        raise ValueError(f"{method.name}: missing continuous score for a target zero")
    if method.value is not None:
        scores["depth_normalized"] = method.value[cells, g] / library[cells] * 1e4
    return scores


def evaluate_screen(dataset: str, args: argparse.Namespace, mixscape: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    inputs = screen_inputs(dataset, args)
    screen = load_screen(dataset, inputs.deploy, inputs.splits, inputs.prepared)
    targets = eligible_targets(screen, args.min_detection)
    if not targets:
        return {}
    library = screen.recorded.sum(axis=1, dtype=np.float64)
    log_library = np.log(library)
    gene_columns = np.array([item.g for item in targets])
    methods = build_methods(screen, gene_columns, inputs, args.seed)
    test, control = screen.test, screen.control

    target_rows, auroc_rows, fill_rows, effect_rows, mixscape_rows = [], [], [], [], []
    filled = {name: {pct: method.fill_source(pct) for pct in EFFECT_FRACTIONS} for name, method in methods.items()}
    recorded_columns = screen.recorded[:, gene_columns].astype(np.float64)
    classes = None
    if mixscape is not None:
        classes = mixscape.set_index("cell_id").reindex(screen.cell_ids)["mixscape_class_global"].astype(str).to_numpy()

    def effect(sums: np.ndarray, column: np.ndarray, gene: str) -> float:
        cp10k = column / sums * 1e4
        return mean_log2_ratio(cp10k[test & (screen.target == gene)], cp10k[test & control])

    for j, item in enumerate(targets):
        labels = likely_dropout_labels(screen, item)
        cells = item.cells
        lib_control, lib_perturbed = library[item.control_zero], library[item.perturbed_zero]
        perturbed_cells = test & (screen.target == item.gene)
        recorded_effect = effect(library, recorded_columns[:, j], item.gene)
        target_rows.append({
            "dataset": dataset, "direction": screen.direction, "gene": item.gene,
            "development_log2fc": item.development_log2fc, "development_detection": item.development_detection,
            "n_control_zero": len(item.control_zero), "n_perturbed_zero": len(item.perturbed_zero),
            "n_control_cells": int((test & control).sum()), "n_perturbed_cells": int(perturbed_cells.sum()),
            "heldout_log2fc_recorded": recorded_effect,
            "library_median_control_zero": float(np.median(lib_control)),
            "library_median_perturbed_zero": float(np.median(lib_perturbed)),
            "library_median_control_cells": float(np.median(library[test & control])),
            "library_median_perturbed_cells": float(np.median(library[perturbed_cells])),
            "library_auroc": auroc(labels, log_library[cells]),
            "library_mannwhitney_p": float(stats.mannwhitneyu(lib_control, lib_perturbed, alternative="two-sided").pvalue),
        })
        for name, method in methods.items():
            for score_type, score in target_scores(method, item, library).items():
                strata_value, pair_share = stratified_auroc(labels, score, log_library[cells], args.depth_strata)
                matched_value, matched_distance = matched_auroc(labels, score, log_library[cells])
                auroc_rows.append({
                    "dataset": dataset, "gene": item.gene, "method": name, "label_aware": name in LABEL_METHODS, "score_type": score_type,
                    "none": auroc(labels, score), "depth_strata": strata_value, "depth_matched": matched_value,
                    "strata_pair_share": pair_share, "matched_median_abs_log_library_difference": matched_distance,
                })
            order = method.order[cells, item.g]
            for pct in FRACTIONS:
                fill_rows.append({
                    "dataset": dataset, "gene": item.gene, "method": name, "fill_pct": pct,
                    "fill_control": float((order[: len(item.control_zero)] <= pct).mean()),
                    "fill_perturbed": float((order[len(item.control_zero):] <= pct).mean()),
                })
            for pct in EFFECT_FRACTIONS:
                sums, columns = filled[name][pct]
                value = effect(sums, columns[:, j], item.gene)
                effect_rows.append({"dataset": dataset, "gene": item.gene, "method": name, "fill_pct": pct,
                                    "log2fc_recorded": recorded_effect, "log2fc_filled": value, "shift": value - recorded_effect})
            if classes is not None:
                perturbed_order = order[len(item.control_zero):]
                perturbed_class = classes[item.perturbed_zero]
                continuous = method.score[cells, item.g].astype(np.float64)
                for label in ("KO", "NP"):
                    members = perturbed_class == label
                    row = {"dataset": dataset, "gene": item.gene, "method": name, "mixscape_class": label, "n_zeros": int(members.sum())}
                    for pct in EFFECT_FRACTIONS:
                        row[f"filled_{pct}pct"] = int((perturbed_order[members] <= pct).sum())
                    if members.any():
                        keep = np.concatenate([np.ones(len(item.control_zero), dtype=bool), members])
                        row["continuous_auroc_vs_control_zeros"] = auroc(labels[keep], continuous[keep])
                    mixscape_rows.append(row)
    return {
        "targets": pd.DataFrame(target_rows),
        "auroc": pd.DataFrame(auroc_rows),
        "fills": pd.DataFrame(fill_rows),
        "effects": pd.DataFrame(effect_rows),
        "mixscape": pd.DataFrame(mixscape_rows),
    }


def interval(values: np.ndarray, draws: np.ndarray) -> list[float]:
    boot = np.nanmean(values[draws], axis=1)
    return [round(float(np.nanmean(values)), 4), round(float(np.nanpercentile(boot, 2.5)), 4), round(float(np.nanpercentile(boot, 97.5)), 4)]


def permutation_p(first: np.ndarray, second: np.ndarray) -> float:
    values = np.concatenate([first, second])
    total, k = values.sum(), len(second)
    combos = np.fromiter(itertools.chain.from_iterable(itertools.combinations(range(len(values)), k)), dtype=np.int16).reshape(-1, k)
    sums = values[combos].sum(axis=1)
    differences = (total - sums) / len(first) - sums / k
    observed = first.mean() - second.mean()
    return float(np.mean(np.abs(differences) >= abs(observed) - 1e-12))


def summarize(tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict:
    targets, aurocs, effects = tables["targets"], tables["auroc"], tables["effects"]
    def scope_draws(scope: str, n: int) -> np.ndarray:
        return np.random.default_rng([args.seed, SCOPES.index(scope)]).integers(0, n, size=(args.draws, n))

    draws = {dataset: scope_draws(dataset, n) for dataset, n in targets.groupby("dataset", sort=False).size().items()}
    report: dict = {"targets": targets.groupby("dataset", sort=False).size().to_dict(), "depth_strata": args.depth_strata, "screens": {}}
    knockdown_targets = targets[targets.direction == "knockdown"]
    draws["knockdown_pooled"] = scope_draws("knockdown_pooled", len(knockdown_targets))
    scopes = [(dataset, [dataset]) for dataset in targets["dataset"].unique()] + [("knockdown_pooled", list(knockdown_targets["dataset"].unique()))]
    for scope, datasets in scopes:
        sub_targets = targets[targets.dataset.isin(datasets)]
        d = draws[scope]
        entry = {
            "n_targets": int(len(sub_targets)),
            "library_auroc": interval(sub_targets["library_auroc"].to_numpy(), d),
            "targets_with_shallower_control_zeros": int((sub_targets["library_median_control_zero"] < sub_targets["library_median_perturbed_zero"]).sum()),
            "median_library_control_zero_over_perturbed_zero": round(float(np.median(sub_targets["library_median_control_zero"] / sub_targets["library_median_perturbed_zero"])), 4),
            "targets_with_library_p_below_0.05": int((sub_targets["library_mannwhitney_p"] < 0.05).sum()),
            "heldout_log2fc_recorded": interval(sub_targets["heldout_log2fc_recorded"].to_numpy(), d),
            "auroc": {},
            "adjusted_minus_unadjusted": {},
            "knockdown_effect_shift": {},
        }
        genes = sub_targets.set_index(["dataset", "gene"]).index
        for (method, score_type), frame in aurocs[aurocs.dataset.isin(datasets)].groupby(["method", "score_type"], sort=False):
            frame = frame.set_index(["dataset", "gene"]).loc[genes]
            key = f"{method}|{score_type}"
            entry["auroc"][key] = {adjustment: interval(frame[adjustment].to_numpy(), d) for adjustment in ADJUSTMENTS}
            entry["auroc"][key]["strata_pair_share"] = round(float(frame["strata_pair_share"].mean()), 4)
            entry["adjusted_minus_unadjusted"][key] = {
                adjustment: interval((frame[adjustment] - frame["none"]).to_numpy(), d) for adjustment in ADJUSTMENTS[1:]
            }
        for (method, pct), frame in effects[effects.dataset.isin(datasets)].groupby(["method", "fill_pct"], sort=False):
            frame = frame.set_index(["dataset", "gene"]).loc[genes]
            entry["knockdown_effect_shift"][f"{method}|{pct}"] = {
                "log2fc_filled": interval(frame["log2fc_filled"].to_numpy(), d),
                "shift": interval(frame["shift"].to_numpy(), d),
            }
        report["screens"][scope] = entry

    first, second = KNOCKDOWN_COMPARISON
    if {first, second} <= set(targets["dataset"]):
        comparison = {}
        for (method, score_type), frame in aurocs.groupby(["method", "score_type"], sort=False):
            a = frame[frame.dataset == first].set_index("gene")
            b = frame[frame.dataset == second].set_index("gene")
            key = f"{method}|{score_type}"
            comparison[key] = {}
            for adjustment in ADJUSTMENTS:
                va, vb = a[adjustment].to_numpy(), b[adjustment].to_numpy()
                boot = np.nanmean(va[draws[first]], axis=1) - np.nanmean(vb[draws[second]], axis=1)
                comparison[key][adjustment] = {
                    "difference": round(float(np.nanmean(va) - np.nanmean(vb)), 4),
                    "ci": [round(float(np.nanpercentile(boot, 2.5)), 4), round(float(np.nanpercentile(boot, 97.5)), 4)],
                    "permutation_p": round(permutation_p(va, vb), 5) if not (np.isnan(va).any() or np.isnan(vb).any()) else None,
                }
        report[f"{first}_minus_{second}"] = comparison

    if len(tables["mixscape"]):
        pooled = tables["mixscape"].groupby(["method", "mixscape_class"], sort=False).agg(
            n_zeros=("n_zeros", "sum"), **{f"filled_{pct}pct": (f"filled_{pct}pct", "sum") for pct in EFFECT_FRACTIONS},
            continuous_auroc_mean=("continuous_auroc_vs_control_zeros", "mean"),
            targets_with_zeros=("continuous_auroc_vs_control_zeros", "count"),
        )
        for pct in EFFECT_FRACTIONS:
            pooled[f"fill_rate_{pct}pct"] = (pooled[f"filled_{pct}pct"] / pooled["n_zeros"]).round(4)
        report["mixscape_papalexi"] = {f"{m}|{c}": row.to_dict() for (m, c), row in pooled.iterrows()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-root", default="artifacts/paper_evidence/downstream_deployment")
    parser.add_argument("--review-root", default="artifacts/paper_evidence/review_round2/knockdown")
    parser.add_argument("--norman-benchmark", default="artifacts/paper_evidence/review_round2/leakage_free/norman_crispra")
    parser.add_argument("--norman-root", default="artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence/review_round2/knockdown/evaluation")
    parser.add_argument("--mixscape", default="artifacts/paper_evidence/review_round2/knockdown/mixscape/papalexi_mixscape_classes.parquet")
    parser.add_argument("--min-detection", type=float, default=0.2)
    parser.add_argument("--depth-strata", type=int, default=5)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    mixscape_path = Path(args.mixscape)
    mixscape = pd.read_parquet(mixscape_path) if mixscape_path.exists() else None
    collected: dict[str, list[pd.DataFrame]] = {}
    for dataset in SCREENS:
        result = evaluate_screen(dataset, args, mixscape if dataset == "papalexi_eccite" else None)
        for name, frame in result.items():
            collected.setdefault(name, []).append(frame)
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in collected.items()}

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    wide = tables["auroc"].pivot_table(index=["dataset", "gene"], columns=["score_type", "method"], values="none")
    wide.columns = [f"auroc_{score}_{method}" for score, method in wide.columns]
    per_target = tables["targets"].set_index(["dataset", "gene"]).join(wide).reset_index()
    ordered = [f"auroc_{score}_{method}" for score in SCORE_TYPES for method in METHOD_ORDER if f"auroc_{score}_{method}" in per_target]
    per_target = per_target[[c for c in per_target.columns if not c.startswith("auroc_")] + ordered]
    per_target.to_csv(output / "per_target_table.csv", index=False, float_format="%.4f")
    report = summarize(tables, args)
    report["mixscape_available"] = mixscape is not None
    (output / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({scope: {key: value["none"] for key, value in entry["auroc"].items() if key.endswith("|continuous")}
                      for scope, entry in report["screens"].items()}, indent=1))


if __name__ == "__main__":
    main()
