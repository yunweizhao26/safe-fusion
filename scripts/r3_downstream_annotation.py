#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from disease_control_annotation import SEED_REFERENCE, leiden, reference_mapping
from disease_control_common import DRAWS, FRACTIONS, SEED, TISSUES, interval, load_heldout, stratified_draws
from evaluate_colon_donor_biology import macro_f1
from r3_downstream_disease import contract, filled, method_list
from safefusion_benchmark.downstream import adjusted_rand_index

REPOSITORY = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", choices=sorted(TISSUES), required=True)
    parser.add_argument("--root", type=Path, default=REPOSITORY / "artifacts/paper_evidence/review_round3/downstream")
    parser.add_argument("--comparators-only", action="store_true")
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    spec = TISSUES[args.tissue]

    data = load_heldout(args.tissue)
    cells = data.cells
    labels = cells["cell_type"].to_numpy()
    donors = sorted(cells["donor"].unique())
    donor_code = pd.Categorical(cells["donor"], categories=donors).codes
    donor_condition = cells.groupby("donor")["condition_label"].first().loc[donors].to_numpy()
    draws = stratified_draws(donor_condition, args.draws, args.seed)
    weights = np.stack([np.bincount(draw, minlength=len(donors))[donor_code] for draw in draws])

    methods = method_list(spec, args.root, args.comparators_only)
    keys = [("unfilled", 0.0), (SEED_REFERENCE, 0.0)] + [(method, fraction) for method in methods for fraction in FRACTIONS]
    assignments, clusters = {}, {}
    for key in keys:
        if key[0] in ("unfilled", SEED_REFERENCE):
            matrix = data.recorded
        else:
            matrix = data.recorded.copy()
            matrix[data.heldout] = filled(data, lambda unit, key=key: contract(unit, *key, args.root))
        seed = args.seed + 1 if key[0] == SEED_REFERENCE else args.seed
        assignments[key] = reference_mapping(data, matrix, seed)
        clusters[key] = leiden(matrix[data.heldout], seed)

    raw_assigned = assignments[("unfilled", 0.0)]
    raw_clusters = clusters[("unfilled", 0.0)]
    rows = []
    for key in keys[1:]:
        changed = (assignments[key] != raw_assigned).astype(float)
        boot = weights @ changed / weights.sum(axis=1)
        boot_ari = [
            adjusted_rand_index(raw_clusters[np.repeat(np.arange(len(labels)), w)], clusters[key][np.repeat(np.arange(len(labels)), w)])
            for w in weights
        ]
        low, high = interval(boot)
        ari_low, ari_high = interval(np.asarray(boot_ari))
        rows.append({
            "method": key[0], "fraction": key[1],
            "changed_share": float(changed.mean()), "changed_share_ci_low": low, "changed_share_ci_high": high,
            "macro_f1_unfilled": macro_f1(labels, raw_assigned), "macro_f1_filled": macro_f1(labels, assignments[key]),
            "leiden_ari_unfilled_vs_filled": adjusted_rand_index(raw_clusters, clusters[key]),
            "leiden_ari_ci_low": ari_low, "leiden_ari_ci_high": ari_high,
        })
    output = args.root / "deployment" / ("annotation_comparators" if args.comparators_only else "annotation") / args.tissue
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "overall.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
