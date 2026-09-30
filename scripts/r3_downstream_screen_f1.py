#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE = REPOSITORY / "artifacts" / "paper_evidence"
SCREENS = {
    "norman_crispra": "Norman CRISPRa (rebuilt)",
    "adamson_crispri": "Adamson CRISPRi",
    "dixit_ko": "Dixit knockout",
    "papalexi_eccite": "Papalexi ECCITE-seq",
}
METHODS = {
    "Safe Fusion": ("safe_fusion", "safe_fusion_condition", "safe_fusion_condition_minus_safe_fusion"),
    "Weighted kNN": ("weighted_knn", "knn_condition", "knn_condition_minus_weighted_knn"),
    "scVI teacher": ("scvi", "scvi_condition", "scvi_condition_minus_scvi"),
}

def formatted(values: list[float]) -> str:
    return f"{values[0]:.2f} [{values[1]:.2f}, {values[2]:.2f}]"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=EVIDENCE / "review_round2" / "knockdown" / "masked_f1" / "masked_f1_by_screen.json")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE / "review_round3" / "downstream" / "screen_f1")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    rows = []
    for screen, label in SCREENS.items():
        entry = summary[screen]
        for method, (unlabeled, labeled, difference) in METHODS.items():
            without = entry["mean_f1_percent"][unlabeled]
            with_labels = entry["mean_f1_percent"][labeled]
            change = entry["label_minus_unlabeled_pp"][difference]
            rows.append({
                "screen": label, "method": method, "n_units": entry["n_units"],
                "without_labels": without[0], "without_labels_lower": without[1], "without_labels_upper": without[2],
                "with_labels": with_labels[0], "with_labels_lower": with_labels[1], "with_labels_upper": with_labels[2],
                "labels_minus_none_pp": change[0], "labels_minus_none_lower": change[1], "labels_minus_none_upper": change[2],
                "interval_excludes_zero": bool(change[1] > 0 or change[2] < 0),
                "without_labels_text": formatted(without), "with_labels_text": formatted(with_labels),
                "labels_minus_none_text": formatted(change),
            })
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "screen_masked_f1.csv", index=False, float_format="%.3f")
    lines = [
        "| Screen | Units | Method | Without labels | With labels | Labels minus none (pp) |",
        "|---|---|---|---|---|---|",
        *(f"| {row.screen} | {row.n_units} | {row.method} | {row.without_labels_text} | {row.with_labels_text} | "
          f"{row.labels_minus_none_text} |" for row in table.itertuples()),
    ]
    (args.output_dir / "screen_masked_f1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
