#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(source, backed="r")
    required = {"donor", "condition"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"missing required observation columns: {sorted(missing)}")

    donor_condition = adata.obs[["donor", "condition"]].astype(str).drop_duplicates()
    if donor_condition["donor"].duplicated().any():
        raise ValueError("each donor must have exactly one condition")

    rng = np.random.default_rng(args.seed)
    assignment: dict[str, int] = {}
    for condition, block in donor_condition.groupby("condition", sort=True, observed=True):
        donors = np.asarray(sorted(block["donor"].tolist()), dtype=object)
        donors = donors[rng.permutation(len(donors))]
        for offset, donor in enumerate(donors):
            assignment[str(donor)] = int(offset % args.folds)

    manifest_folds: list[dict] = []
    cell_ids = adata.obs_names.astype(str).to_numpy()
    donors = adata.obs["donor"].astype(str).to_numpy()
    conditions = adata.obs["condition"].astype(str).to_numpy()
    for fold in range(args.folds):
        heldout = {donor for donor, assigned in assignment.items() if assigned == fold}
        split = np.where(np.isin(donors, sorted(heldout)), "test", "development")
        frame = pd.DataFrame({
            "cell_id": cell_ids,
            "biological_unit": donors,
            "condition": conditions,
            "split": split,
            "fold": fold,
        })
        path = destination / f"fold_{fold}" / "splits.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        heldout_table = (
            donor_condition[donor_condition["donor"].isin(heldout)]
            .sort_values(["condition", "donor"])
            .to_dict(orient="records")
        )
        manifest_folds.append({
            "fold": fold,
            "split": str(path),
            "split_sha256": sha256(path),
            "test_donors": heldout_table,
            "test_cells": int(np.sum(split == "test")),
            "training_cells": int(np.sum(split == "development")),
        })

    if set(assignment) != set(donor_condition["donor"]):
        raise AssertionError("not every donor was assigned")
    if any(sum(donor in {row["donor"] for row in fold["test_donors"]} for fold in manifest_folds) != 1 for donor in assignment):
        raise AssertionError("each donor must be held out exactly once")

    manifest = {
        "input": str(source),
        "input_sha256": sha256(source),
        "seed": args.seed,
        "folds": args.folds,
        "unit": "donor",
        "stratification": "condition",
        "condition_counts": donor_condition["condition"].value_counts().sort_index().to_dict(),
        "fold_details": manifest_folds,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
