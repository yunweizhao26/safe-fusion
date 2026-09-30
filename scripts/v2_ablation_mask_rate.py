#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

from safefusion_benchmark.corruption import _quantile_bins
from safefusion_benchmark.hashing import sha256_file
from v2_ablation_common import ABLATION_ROOT, UNIT_KEYS, mask_rate_unit, units

BINS = 4

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", choices=UNIT_KEYS, required=True)
    parser.add_argument("--mask-rate", type=float, required=True)
    parser.add_argument("--ablation-root", type=Path, default=ABLATION_ROOT)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    unit = units()[args.unit]
    destination = mask_rate_unit(unit, args.ablation_root, args.mask_rate)
    output = Path(destination.corrupted).parent
    corrupted = ad.read_h5ad(unit.corrupted)
    truth = ad.read_h5ad(unit.truth)
    if not (np.array_equal(corrupted.obs_names, truth.obs_names) and np.array_equal(corrupted.var_names, truth.var_names)):
        raise ValueError("truth and masked input differ in cell or gene order")
    recorded = dense(truth.layers["counts"]).astype(np.int64)
    benchmark = dense(corrupted.layers["corrupted_counts"]).astype(np.int64)
    coordinates = pd.read_parquet(unit.coordinates)
    cell_ids = corrupted.obs_names.astype(str).to_numpy()
    gene_ids = corrupted.var_names.astype(str).to_numpy()
    split = pd.read_parquet(unit.splits).set_index("cell_id").loc[cell_ids, "split"].astype(str).to_numpy()
    training = split != "test"

    rows = coordinates["cell_index"].to_numpy(dtype=np.int64)
    cols = coordinates["gene_index"].to_numpy(dtype=np.int64)
    restored = benchmark.copy()
    restored[rows, cols] = recorded[rows, cols]
    if not np.array_equal(restored, recorded):
        raise ValueError("the benchmark mask does not reproduce the recorded counts")

    gene_bins = _quantile_bins(recorded.mean(axis=0), BINS)
    library_bins = _quantile_bins(recorded.sum(axis=1), BINS)
    stratum_of = library_bins[:, None] * BINS + gene_bins[None, :]
    is_benchmark_masked = np.zeros(recorded.shape, dtype=bool)
    is_benchmark_masked[rows, cols] = True

    rng = np.random.default_rng(args.seed)
    chosen_rows, chosen_cols, strata_report = [], [], []
    candidate_rows, candidate_cols = np.where((recorded > 0) & training[:, None])
    candidate_strata = stratum_of[candidate_rows, candidate_cols]
    for stratum in np.unique(candidate_strata):
        members = np.flatnonzero(candidate_strata == stratum)
        in_benchmark = is_benchmark_masked[candidate_rows[members], candidate_cols[members]]
        masked_members, open_members = members[in_benchmark], members[~in_benchmark]
        target = max(1, int(round(args.mask_rate * len(members))))
        if target <= len(masked_members):
            keep = np.sort(rng.choice(masked_members, size=target, replace=False))
        else:
            keep = np.sort(np.concatenate([masked_members, rng.choice(open_members, size=target - len(masked_members), replace=False)]))
        chosen_rows.append(candidate_rows[keep])
        chosen_cols.append(candidate_cols[keep])
        strata_report.append({
            "stratum": int(stratum), "library_quartile": int(stratum // BINS), "gene_quartile": int(stratum % BINS),
            "training_nonzeros": int(len(members)), "benchmark_masked": int(len(masked_members)), "masked": int(target),
        })
    train_rows = np.concatenate(chosen_rows)
    train_cols = np.concatenate(chosen_cols)
    order = np.lexsort((train_cols, train_rows))
    train_rows, train_cols = train_rows[order], train_cols[order]

    counts = recorded.copy()
    counts[train_rows, train_cols] = 0
    test_coordinates = coordinates.loc[~training[rows]].copy()
    counts[test_coordinates["cell_index"].to_numpy(dtype=np.int64), test_coordinates["gene_index"].to_numpy(dtype=np.int64)] = 0
    if not np.array_equal(counts[~training], benchmark[~training]):
        raise AssertionError("the test cells differ from the benchmark")

    units_column = truth.obs[unit.unit_column].astype(str).to_numpy()
    training_coordinates = pd.DataFrame({
        "cell_id": cell_ids[train_rows],
        "gene_id": gene_ids[train_cols],
        "cell_index": train_rows,
        "gene_index": train_cols,
        "corruption_type": "stratified_nonzero_mask",
        "original_value": recorded[train_rows, train_cols].astype(np.float32),
        "corrupted_value": np.zeros(len(train_rows), dtype=np.float32),
        "biological_unit": units_column[train_rows],
        "split": split[train_rows],
    })
    new_coordinates = pd.concat([training_coordinates, test_coordinates[training_coordinates.columns]], ignore_index=True)
    if new_coordinates.duplicated(["cell_index", "gene_index"]).any():
        raise AssertionError("coordinates are not unique")

    output.mkdir(parents=True, exist_ok=True)
    matrix = sparse.csr_matrix(counts.astype(np.int32))
    result = ad.AnnData(X=matrix, obs=corrupted.obs.copy(), var=corrupted.var.copy())
    result.layers["corrupted_counts"] = matrix.copy()
    design = {
        "design": "training cells masked at mask_rate within the 16 benchmark strata, nested with the benchmark mask; "
                  "test cells as in the benchmark",
        "mask_rate": args.mask_rate, "seed": args.seed, "benchmark": str(unit.corrupted),
    }
    result.uns["corruption"] = json.dumps(design)
    result.write_h5ad(destination.corrupted)
    new_coordinates.to_parquet(destination.coordinates, index=False)
    benchmark_training = int(training[rows].sum())
    manifest = {
        **design,
        "unit": unit.key,
        "benchmark_sha256": sha256_file(unit.corrupted),
        "training_nonzeros": int(len(candidate_rows)),
        "training_masked": int(len(train_rows)),
        "training_mask_rate": float(len(train_rows) / len(candidate_rows)),
        "benchmark_training_masked": benchmark_training,
        "benchmark_training_mask_rate": float(benchmark_training / len(candidate_rows)),
        "training_masked_kept_from_benchmark": int(is_benchmark_masked[train_rows, train_cols].sum()),
        "test_masked": int(len(test_coordinates)),
        "strata": strata_report,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items() if key != "strata"}, indent=2))

if __name__ == "__main__":
    main()
