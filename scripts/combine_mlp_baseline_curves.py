#!/usr/bin/env python3
"""Combine the MLP selector coverage curves with matched baseline curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts" / "paper_evidence"


def load_curve(path: Path) -> tuple[list[dict[str, float]], int, int]:
    report = json.loads(path.read_text())
    return (
        report["exact_budget_curve"],
        int(report["test"]["n_masked_positives"]),
        int(report["test"]["n_zeros"]),
    )


def pool_curves(paths: list[Path]) -> pd.DataFrame:
    loaded = [load_curve(path) for path in paths]
    point_count = len(loaded[0][0])
    if point_count != 1000 or any(len(curve) != point_count for curve, _, _ in loaded):
        raise ValueError("MLP coverage curves must each contain 1000 points")
    rows = []
    for index in range(point_count):
        points = [curve[index] for curve, _, _ in loaded]
        requested = float(points[0]["requested_fill_fraction"])
        if any(abs(float(point["requested_fill_fraction"]) - requested) > 1e-12 for point in points):
            raise ValueError("MLP coverage grids differ")
        selected = int(sum(int(point["n_selected"]) for point in points))
        true_positive = int(sum(int(point["n_true_positive"]) for point in points))
        positives = int(sum(value[1] for value in loaded))
        zeros = int(sum(value[2] for value in loaded))
        precision = true_positive / selected
        recall = true_positive / positives
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "requested_fill_fraction": requested,
                "realized_fill_fraction": selected / zeros,
                "n_selected": selected,
                "n_true_positive": true_positive,
                "n_masked_positives": positives,
                "n_zeros": zeros,
                "masked_precision": precision,
                "masked_recall": recall,
                "masked_f1": f1,
            }
        )
    return pd.DataFrame(rows)


def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    area = np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return float(area / (x[-1] - x[0]))


def summarize(table: pd.DataFrame) -> dict[str, dict]:
    """Margins of the Safe Fusion selector over the best comparator of each family.

    The ``main`` family holds the untrained comparators ranked by their own
    count-scale value (scGCL is supplementary and excluded). The ``stacked``
    family holds the same MLP selector trained on one comparator's value plus
    the context features.
    """
    result: dict[str, dict] = {}
    ranges = ((0.001, 0.01), (0.001, 0.02), (0.001, 0.05), (0.001, 0.10), (0.001, 0.20), (0.001, 1.0))
    for dataset, frame in table.groupby("dataset", sort=False):
        pivot = frame.pivot(index="coverage_index", columns="method", values="masked_f1").sort_index()
        coverage = (
            frame.loc[frame["method"] == "Safe Fusion MLP"]
            .sort_values("coverage_index")["requested_fill_fraction"]
            .to_numpy()
        )
        families = frame.drop_duplicates("method").set_index("method")["comparison"]
        dataset_summary: dict[str, object] = {}
        for family in ("main", "stacked"):
            competitors = [method for method, value in families.items() if value == family]
            if not competitors:
                continue
            best_competitor = pivot[competitors].max(axis=1)
            best_name = pivot[competitors].idxmax(axis=1)
            family_summary: dict[str, object] = {"competitors": competitors}
            for low, high in ranges:
                keep = (coverage >= low - 1e-12) & (coverage <= high + 1e-12)
                x = coverage[keep]
                mlp = pivot.loc[keep, "Safe Fusion MLP"].to_numpy()
                best = best_competitor.loc[keep].to_numpy()
                margin = mlp - best
                family_summary[f"{low:.3f}_{high:.3f}"] = {
                    "points": int(keep.sum()),
                    "fraction_mlp_best": float(np.mean(margin >= 0.0)),
                    "mean_f1_margin": float(np.mean(margin)),
                    "minimum_f1_margin": float(np.min(margin)),
                    "maximum_f1_margin": float(np.max(margin)),
                    "mlp_mean_f1_over_coverage": normalized_auc(x, mlp),
                    "best_competitor_mean_f1_over_coverage": normalized_auc(x, best),
                    "best_competitor_counts": best_name.loc[keep].value_counts().to_dict(),
                }
            dataset_summary[family] = family_summary
        result[str(dataset)] = dataset_summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=EVIDENCE / "selector_f1_fillrate_baselines_1000_points.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EVIDENCE / "selector_f1_fillrate_mlp_baselines_1000_points.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=EVIDENCE / "selector_f1_fillrate_mlp_baselines_summary.json",
    )
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE)
    parser.add_argument(
        "--stacked-root",
        type=Path,
        default=EVIDENCE / "stacked_selector_baselines",
        help="Outputs of stacked_selector_baselines.py; skipped when absent.",
    )
    args = parser.parse_args()

    # The selector runs that also produce every downstream output
    # (scripts/slurm_complete_downstream_selectors.sh).
    evidence = args.evidence_root
    unit_keys = {
        "Pancreas": [f"pancreas_{fold}" for fold in range(3)],
        "Colon": ["colon"],
        "CRISPRa": ["norman_crispra"],
    }
    mlp_paths = {
        "Pancreas": [
            evidence / "pancreas_crossfit" / f"fold_{fold}" / "selector_mlp_biology_range_fullteachers" / "calibration_report.json"
            for fold in range(3)
        ],
        "Colon": [evidence / "selector_mlp_biology_range" / "colon" / "calibration_report.json"],
        "CRISPRa": [evidence / "selector_mlp_biology_range_fullteachers" / "norman_crispra" / "calibration_report.json"],
    }
    frames = []
    for dataset, paths in mlp_paths.items():
        frame = pool_curves(paths)
        frame.insert(0, "comparison", "safe_fusion")
        frame.insert(0, "method", "Safe Fusion MLP")
        frame.insert(0, "dataset", dataset)
        frames.append(frame)
        # Stacked baselines: the same MLP selector trained on one comparator's
        # count-scale value plus the context features.
        if args.stacked_root.exists():
            first_unit = args.stacked_root / unit_keys[dataset][0]
            for method_dir in sorted(path for path in first_unit.iterdir() if (path / "report.json").exists()):
                reports = [args.stacked_root / key / method_dir.name / "report.json" for key in unit_keys[dataset]]
                name = json.loads(reports[0].read_text())["comparator"]
                stacked = pool_curves(reports)
                stacked.insert(0, "comparison", "stacked")
                stacked.insert(0, "method", f"{name} (stacked)")
                stacked.insert(0, "dataset", dataset)
                frames.append(stacked)

    baseline = pd.read_csv(args.baseline)
    baseline = baseline.loc[baseline["method"] != "Safe Fusion"]
    combined = pd.concat([*frames, baseline], ignore_index=True)
    canonical_coverage = np.linspace(0.001, 1.0, 1000)
    normalized = []
    for (dataset, method), frame in combined.groupby(["dataset", "method"], sort=False):
        frame = frame.sort_values("requested_fill_fraction").reset_index(drop=True)
        if len(frame) != len(canonical_coverage):
            raise ValueError(f"{dataset} {method} does not contain 1000 coverage points")
        observed = frame["requested_fill_fraction"].to_numpy()
        if not np.allclose(observed, canonical_coverage, rtol=0.0, atol=1e-12):
            raise ValueError(f"{dataset} {method} uses an unexpected coverage grid")
        frame["coverage_index"] = np.arange(len(frame))
        frame["requested_fill_fraction"] = canonical_coverage
        normalized.append(frame)
    combined = pd.concat(normalized, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    args.summary_output.write_text(json.dumps(summarize(combined), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
