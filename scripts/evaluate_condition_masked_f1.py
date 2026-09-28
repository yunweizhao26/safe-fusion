#!/usr/bin/env python3
"""Masked F1 on the four perturbation screens, with and without perturbation labels.

The candidates of each masked screen benchmark (Norman, Adamson, Dixit and
Papalexi) are the zeros of the held-out cells in the masked input, and the
positives are the masked entries among them. At each fill fraction b from 1% to
10%, a comparator fills the top round(b * |Z|) candidates ranked by its own
value on the count scale (masked_f1_units.count_scale_values, ties broken at
random), and Safe Fusion fills the entries of its selector output contract at
b. Masked F1 is 2 TP / (filled + masked positives). The report gives, for every
method, the mean over the ten fill fractions in each screen and the average of
these means over the four screens.

Label methods use the perturbation labels (scripts/slurm_condition_aware_screens.sh):
the weighted kNN teacher borrows only from cells with the same perturbation, the
scVI teacher conditions on it, and the Safe Fusion selector adds the gene's
mean and zero fraction within the cell's perturbation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from compute_matched_baseline_f1_curves import tie_broken_order
from masked_f1_units import EVIDENCE, ROOT, UNIT_FRACTIONS, Unit, count_scale_values, fraction_name, load_unit

SCREENS = ("norman_crispra", "adamson_crispri", "dixit_ko", "papalexi_eccite")
# Comparator contracts, relative to the screen's methods directory. Each ranks
# the candidates by its own imputed value.
VALUE_METHODS = {
    "svd": "svd_impute",
    "weighted_knn": "graph_smooth",
    "magic": "magic_inductive",
    "scvi": "scvi_inductive",
    "knn_condition": "graph_smooth_condition",
    "scvi_condition": "scvi_inductive_condition",
}
SELECTOR_METHODS = ("safe_fusion", "safe_fusion_condition")
METHOD_ORDER = ("safe_fusion", "svd", "weighted_knn", "magic", "scvi",
                "safe_fusion_condition", "knn_condition", "scvi_condition")


def screen_unit(dataset: str, evidence: Path, external: Path, seed: int) -> tuple[Unit, dict[str, Path]]:
    """Masked benchmark paths of one screen and the selector directory of each Safe Fusion variant."""
    if dataset == "norman_crispra":
        root = evidence / "norman_crispra"
        methods = root / "methods"
        selectors = {
            "safe_fusion": evidence / "selector_mlp_biology_range_fullteachers" / "norman_crispra",
            "safe_fusion_condition": methods / "selector_condition",
        }
        truth = ROOT / "external_data" / "prepared" / "norman_crispra.h5ad"
    else:
        root = methods = external / dataset
        selectors = {
            "safe_fusion": root / "selector_mlp_biology_range",
            "safe_fusion_condition": root / "selector_condition",
        }
        truth = ROOT / "external_data" / "prepared" / f"{dataset}.h5ad"
    unit = Unit(
        dataset=dataset,
        key=dataset,
        corrupted=root / "corrupted.h5ad",
        coordinates=root / "coordinates.parquet",
        splits=root / "splits.parquet",
        truth=truth,
        fit_split="development",
        unit_column="target",
        tie_seed=seed,
        selector_dir=selectors["safe_fusion"],
        contracts={name: methods / contract for name, contract in VALUE_METHODS.items()},
    )
    return unit, selectors


def evaluate_screen(unit: Unit, selectors: dict[str, Path]) -> list[dict]:
    data = load_unit(unit)
    test_cells = np.flatnonzero(data.split == "test")
    local_rows, cols = np.where(data.counts[test_cells] == 0)
    rows = test_cells[local_rows]
    labels = data.masked[rows, cols]
    n_zeros, positives = len(labels), int(labels.sum())

    selections: dict[str, dict[float, np.ndarray]] = {}
    for name, contract in unit.contracts.items():
        scores, _ = count_scale_values(contract, data, rows, cols)
        rank = np.empty(n_zeros, dtype=np.int64)
        rank[tie_broken_order(scores, unit.tie_seed)] = np.arange(n_zeros)
        selections[name] = {fraction: rank < max(1, int(round(fraction * n_zeros))) for fraction in UNIT_FRACTIONS}
    for name in SELECTOR_METHODS:
        selections[name] = {
            fraction: np.asarray(
                np.load(selectors[name] / f"safe_fusion_calibrated_mlp_topk_{fraction_name(fraction)}" / "mean.npy", mmap_mode="r")[rows, cols]
            ) != 0
            for fraction in UNIT_FRACTIONS
        }

    records = []
    for name in METHOD_ORDER:
        for fraction, selected in selections[name].items():
            true_positive = int((selected & labels).sum())
            records.append({
                "dataset": unit.dataset,
                "method": name,
                "fill_fraction": fraction,
                "n_zeros": n_zeros,
                "n_masked_positives": positives,
                "n_selected": int(selected.sum()),
                "n_true_positive": true_positive,
                "masked_f1": 2.0 * true_positive / (int(selected.sum()) + positives),
            })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE)
    parser.add_argument("--external-root", type=Path, default=ROOT / "artifacts" / "external_perturbseq")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE / "perturbation_zeros")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    records = []
    for dataset in SCREENS:
        unit, selectors = screen_unit(dataset, args.evidence_root, args.external_root, args.seed)
        records.extend(evaluate_screen(unit, selectors))
    frame = pd.DataFrame(records)
    by_screen = frame.groupby(["method", "dataset"], sort=False)["masked_f1"].mean().unstack("dataset")[list(SCREENS)]
    by_screen = by_screen.loc[list(METHOD_ORDER)]
    report = {
        "fill_fractions": list(UNIT_FRACTIONS),
        "masked_f1_mean_over_fractions_by_screen": by_screen.round(6).to_dict(orient="index"),
        "masked_f1_mean_over_screens": by_screen.mean(axis=1).round(6).to_dict(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "masked_f1_by_fraction.csv", index=False)
    (args.output_dir / "masked_f1_report.json").write_text(json.dumps(report, indent=1) + "\n")
    table = (100 * by_screen).assign(mean_over_screens=100 * by_screen.mean(axis=1))
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
