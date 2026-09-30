#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from masked_f1_units import (
    EVIDENCE,
    NATIVE_SCALES,
    STACKED_COMPARATORS,
    UnitData,
    add_root_arguments,
    count_scale_values,
    load_unit,
    units_from_args,
)
from selector_attribution import Candidates, fit_scores, selector_features, teacher_feature_names
from stacked_selector_baselines import candidate_keys, exact_budget_curve

FUSION_VALUE_ROOT = EVIDENCE / "review_round2" / "fusion_value"
UNIT_KEYS = ("pancreas_0", "pancreas_1", "pancreas_2", "colon", "norman_crispra")

INDUCTIVE_TEACHERS = ("Gene median", "SVD", "Weighted kNN", "MAGIC (inductive)", "scVI (inductive)")

TRANSDUCTIVE_TEACHERS = ("gene_median", "svd_impute", "graph_smooth", "magic", "scvi")

DEFAULT_FAMILIES = ("fusion", "leave_one_out", "architecture", "stacked", "transductive")
FAMILIES = (
    *DEFAULT_FAMILIES, "transductive_leave_one_out", "transductive_architecture", "transductive_feature_group",
)

@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    sources: tuple[str, ...]
    architecture: str

    feature_variant: str = "full"

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")

def variants() -> list[Variant]:
    transductive_sources = tuple(f"transductive:{name}" for name in TRANSDUCTIVE_TEACHERS)
    result = [Variant("Safe Fusion", "fusion", INDUCTIVE_TEACHERS, "mlp")]
    result += [
        Variant(f"Safe Fusion without {teacher}", "leave_one_out",
                tuple(name for name in INDUCTIVE_TEACHERS if name != teacher), "mlp")
        for teacher in INDUCTIVE_TEACHERS
    ]
    result += [
        Variant("Safe Fusion (logistic)", "architecture", INDUCTIVE_TEACHERS, "logistic"),
        Variant("Safe Fusion (gradient boosting)", "architecture", INDUCTIVE_TEACHERS, "hist_gbdt"),
    ]
    result += [Variant(f"{name} (stacked)", "stacked", (name,), "mlp") for name in STACKED_COMPARATORS]
    result.append(Variant("Safe Fusion (transductive)", "transductive", transductive_sources, "mlp"))
    result += [
        Variant(f"Safe Fusion (transductive) without {teacher}", "transductive_leave_one_out",
                tuple(f"transductive:{name}" for name in TRANSDUCTIVE_TEACHERS if name != teacher), "mlp")
        for teacher in TRANSDUCTIVE_TEACHERS
    ]
    result += [
        Variant("Safe Fusion (transductive, logistic)", "transductive_architecture", transductive_sources, "logistic"),
        Variant("Safe Fusion (transductive, gradient boosting)", "transductive_architecture", transductive_sources, "hist_gbdt"),
    ]
    result += [
        Variant("Safe Fusion (transductive), teacher features only", "transductive_feature_group",
                transductive_sources, "mlp", feature_variant="teacher_only"),
        Variant("Safe Fusion (transductive), context features only", "transductive_feature_group",
                transductive_sources, "mlp", feature_variant="context_only"),
    ]
    return result

def source_contract(source: str, unit, transductive_root: Path) -> Path:
    if source.startswith("transductive:"):
        return transductive_root / unit.key / source.split(":", 1)[1]
    return unit.contracts[source]

def source_values(contract: Path, data: UnitData, fit: tuple[np.ndarray, np.ndarray],
                  test: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, str]:
    fit_values, scale = count_scale_values(contract, data, *fit)
    test_values, _ = count_scale_values(contract, data, *test)
    if scale in NATIVE_SCALES:

        offset = float(np.min(fit_values))
        fit_values, test_values = fit_values - offset, test_values - offset
    return fit_values, test_values, scale

def main() -> None:
    parser = argparse.ArgumentParser()
    add_root_arguments(parser)
    parser.add_argument("--unit", help="Unit to fit (with --variants), or use --array-index. One of --unit-keys.")
    parser.add_argument(
        "--unit-keys", nargs="+", default=list(UNIT_KEYS),
        help="Units of the manifest that --unit and --array-index address, in array order (default: the fusion-value units).",
    )
    parser.add_argument("--variants", nargs="+", default=None, help="Variant names (default: every variant of --families).")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(DEFAULT_FAMILIES))
    parser.add_argument(
        "--array-index", type=int, default=None,
        help="Fit one (unit, variant) pair: unit index // number of variants, variant index % number of variants.",
    )
    parser.add_argument("--list-variants", action="store_true", help="Print the variants of --families and exit.")
    parser.add_argument(
        "--unit-paths", action="store_true",
        help="Print the masked input, coordinates, splits and standard MAGIC, scVI and SAVER contracts of --unit and exit.",
    )
    parser.add_argument("--output-root", type=Path, default=FUSION_VALUE_ROOT / "selectors")
    parser.add_argument("--transductive-root", type=Path, default=FUSION_VALUE_ROOT / "transductive")
    parser.add_argument("--max-fit-rows", type=int, default=2_000_000)
    parser.add_argument("--score-batch-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    if args.unit is not None and args.unit not in args.unit_keys:
        parser.error(f"--unit must be one of {args.unit_keys}")
    selected = [variant for variant in variants() if variant.family in args.families]
    if args.variants is not None:
        by_name = {variant.name: variant for variant in variants()}
        unknown = sorted(set(args.variants) - set(by_name))
        if unknown:
            raise ValueError(f"unknown variants: {unknown}")
        selected = [by_name[name] for name in args.variants]
    if args.list_variants:
        print("\n".join(f"{variant.family}\t{variant.name}" for variant in selected))
        print(f"array tasks for these variants: 0-{len(args.unit_keys) * len(selected) - 1}", file=sys.stderr)
        return
    if args.unit_paths:
        unit = {unit.key: unit for unit in units_from_args(args, args.seed)}[args.unit]
        print(" ".join(str(path) for path in (
            unit.corrupted, unit.coordinates, unit.splits,
            unit.contracts["MAGIC"], unit.contracts["scVI"], unit.contracts["SAVER"],
        )))
        return
    if args.array_index is not None:
        unit_key = args.unit_keys[args.array_index // len(selected)]
        selected = [selected[args.array_index % len(selected)]]
    elif args.unit is not None:
        unit_key = args.unit
    else:
        parser.error("give --unit or --array-index")

    unit = {unit.key: unit for unit in units_from_args(args, args.seed)}[unit_key]
    data = load_unit(unit)
    counts = data.counts
    fit_mask = data.split == unit.fit_split
    test_mask = data.split == "test"
    gene_mean = np.log1p(np.mean(counts[fit_mask], axis=0))
    gene_dropout = np.mean(counts[fit_mask] <= 0, axis=0)
    log_library = {
        "calibrated_selective_fill": np.log1p(counts.sum(axis=1)),
        "stacked_selector_baselines": np.log1p(data.library),
    }

    fit_rows, fit_cols = candidate_keys(counts, fit_mask, data.masked, args.max_fit_rows, args.seed)
    test_rows, test_cols = np.where((counts == 0) & test_mask[:, None])
    fit_labels = data.masked[fit_rows, fit_cols].astype(np.int8)
    test_labels = data.masked[test_rows, test_cols].astype(np.int8)

    for variant in selected:
        started = time.perf_counter()
        contracts = [source_contract(source, unit, args.transductive_root) for source in variant.sources]
        fit_stack, test_stack, scales = [], [], []
        for contract in contracts:
            fit_values, test_values, scale = source_values(contract, data, (fit_rows, fit_cols), (test_rows, test_cols))
            fit_stack.append(fit_values)
            test_stack.append(test_values)
            scales.append(scale)
        names = teacher_feature_names(list(variant.sources))
        convention = "stacked_selector_baselines" if variant.family == "stacked" else "calibrated_selective_fill"

        def candidates(rows, cols, labels, stack) -> Candidates:
            placeholder = np.zeros(len(rows), dtype=np.float32)
            values = np.stack(stack)
            return Candidates(
                rows=rows, cols=cols, labels=labels,
                features=selector_features(values, gene_mean[cols], gene_dropout[cols], log_library[convention][rows]),
                fused_values=placeholder, teacher_values=values, truth_values=placeholder,
                feature_names=names,
            )

        fit_score, test_score, model_report = fit_scores(
            variant.feature_variant, variant.architecture,
            candidates(fit_rows, fit_cols, fit_labels, fit_stack),
            candidates(test_rows, test_cols, test_labels, test_stack),
            args.max_fit_rows, args.score_batch_rows, args.seed,
        )
        output = args.output_root / unit.key / variant.slug
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "test_scores.npy", test_score.astype(np.float32), allow_pickle=False)
        report = {
            "variant": variant.name,
            "family": variant.family,
            "architecture": variant.architecture,
            "sources": list(variant.sources),
            "contracts": [str(path) for path in contracts],
            "stored_scales": scales,
            "features": list(names),
            "feature_convention": convention,
            "dataset": unit.dataset,
            "unit": unit.key,
            "fit_split": unit.fit_split,
            unit.fit_split: {
                "n_zeros": int(len(fit_labels)),
                "n_masked_positives": int(fit_labels.sum()),
                "roc_auc": float(roc_auc_score(fit_labels, fit_score)),
                "pr_auc": float(average_precision_score(fit_labels, fit_score)),
            },
            "test": {
                "n_zeros": int(len(test_labels)),
                "n_masked_positives": int(test_labels.sum()),
                "roc_auc": float(roc_auc_score(test_labels, test_score)),
                "pr_auc": float(average_precision_score(test_labels, test_score)),
            },
            "test_candidate_order": "row-major (cell, gene) order of the zeros of the held-out cells",
            "test_labels_used_for_training": False,
            "seed": args.seed,
            "model": model_report,
            "elapsed_seconds": time.perf_counter() - started,
            "exact_budget_curve": exact_budget_curve(test_score, test_labels),
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(json.dumps({
            "unit": unit.key, "variant": variant.name, "test_pr_auc": round(report["test"]["pr_auc"], 5),
            "elapsed_seconds": round(report["elapsed_seconds"], 1),
        }), flush=True)

if __name__ == "__main__":
    main()
