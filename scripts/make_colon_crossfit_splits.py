#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

STATUSES = ("healthy", "crohn_noninflamed", "inflamed")

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def donor_status(obs: pd.DataFrame) -> pd.Series:
    frame = obs[["donor", "disease", "condition"]].astype(str)
    healthy = frame["disease"].eq("normal").groupby(frame["donor"]).all()
    crohn = frame["disease"].ne("normal").groupby(frame["donor"]).all()
    inflamed = frame["condition"].eq("inflamed").groupby(frame["donor"]).any()
    if not (healthy ^ crohn).all():
        raise ValueError("each donor must be either healthy or a Crohn disease donor")
    if (healthy & inflamed).any():
        raise ValueError("a healthy donor has an inflamed sample")
    status = pd.Series("crohn_noninflamed", index=healthy.index, name="disease_status")
    status[healthy] = "healthy"
    status[inflamed] = "inflamed"
    return status.sort_index()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="external_data/prepared/colon_epithelial.h5ad")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence/review_round3/colon_crossfit")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--development-fraction", type=float, default=0.50, help="Production colon split (configs/colon_pilot.yaml).")
    parser.add_argument("--validation-fraction", type=float, default=0.25, help="Production colon split (configs/colon_pilot.yaml).")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    obs = ad.read_h5ad(source, backed="r").obs
    status = donor_status(obs)
    validation_share = args.validation_fraction / (args.development_fraction + args.validation_fraction)

    rng = np.random.default_rng(args.seed)
    test_fold: dict[str, int] = {}
    for name in STATUSES:
        donors = np.asarray(sorted(status.index[status == name]), dtype=object)
        donors = donors[rng.permutation(len(donors))]
        for offset, donor in enumerate(donors):
            test_fold[str(donor)] = int(offset % args.folds)
    if set(test_fold) != set(status.index):
        raise AssertionError("not every donor was assigned")

    cell_ids = obs.index.astype(str).to_numpy()
    donors = obs["donor"].astype(str).to_numpy()
    cell_status = status.reindex(donors).to_numpy()
    fold_details = []
    for fold in range(args.folds):
        assignment: dict[str, str] = {}
        for name in STATUSES:
            training = np.asarray(
                sorted(donor for donor, held in test_fold.items() if held != fold and status[donor] == name), dtype=object
            )
            training = training[rng.permutation(len(training))]
            n_validation = int(round(validation_share * len(training)))
            assignment.update({str(donor): "validation" for donor in training[:n_validation]})
            assignment.update({str(donor): "development" for donor in training[n_validation:]})
        assignment.update({donor: "test" for donor, held in test_fold.items() if held == fold})
        split = np.asarray([assignment[donor] for donor in donors], dtype=object)
        frame = pd.DataFrame({
            "cell_id": cell_ids,
            "biological_unit": donors,
            "disease_status": cell_status,
            "split": split,
            "fold": fold,
        })
        path = destination / f"fold_{fold}" / "splits.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        donor_table = (
            pd.DataFrame({"donor": list(assignment), "split": list(assignment.values())})
            .assign(disease_status=lambda table: status.reindex(table["donor"]).to_numpy())
        )
        fold_details.append({
            "fold": fold,
            "split": str(path),
            "split_sha256": sha256(path),
            "donors": {
                name: {
                    status_name: sorted(block["donor"].tolist())
                    for status_name, block in donor_table.loc[donor_table["split"] == name].groupby("disease_status", sort=True)
                }
                for name in ("development", "validation", "test")
            },
            "donor_counts": donor_table["split"].value_counts().sort_index().to_dict(),
            "cell_counts": pd.Series(split).value_counts().sort_index().to_dict(),
        })

    held_out = sorted(donor for detail in fold_details for block in detail["donors"]["test"].values() for donor in block)
    if held_out != sorted(status.index):
        raise AssertionError("each donor must be held out exactly once")

    manifest = {
        "input": str(source),
        "input_sha256": sha256(source),
        "seed": args.seed,
        "folds": args.folds,
        "unit": "donor",
        "stratification": "disease_status",
        "disease_status_rule": {
            "healthy": "every cell has disease 'normal'",
            "inflamed": "Crohn disease donor with at least one cell from an inflamed sample",
            "crohn_noninflamed": "Crohn disease donor whose cells all come from non-inflamed samples",
        },
        "status_counts": status.value_counts().sort_index().to_dict(),
        "validation_share_of_training_donors": validation_share,
        "training_donor_split": (
            "Within each fold and disease status, the training donors are ordered at random and the first "
            "round(validation_share x n) become validation donors; the rest are development donors."
        ),
        "fold_details": fold_details,
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
