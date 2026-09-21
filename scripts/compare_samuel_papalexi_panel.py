#!/usr/bin/env python3
"""Check the panel emitted by Samuel's script against the independent audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samuel-panel", required=True)
    parser.add_argument("--audit-panel", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samuel = pd.read_csv(args.samuel_panel).rename(columns={"measurement": "samuel_count"})
    audit = pd.read_parquet(args.audit_panel)[["cell_id", "gene", "protein", "adt_count"]]
    merged = samuel.merge(
        audit,
        on=["cell_id", "gene", "protein"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    common = merged["_merge"] == "both"
    report = {
        "samuel_rows": int(len(samuel)),
        "audit_rows": int(len(audit)),
        "common_rows": int(common.sum()),
        "samuel_only_rows": int((merged["_merge"] == "left_only").sum()),
        "audit_only_rows": int((merged["_merge"] == "right_only").sum()),
        "exact_count_match": bool(
            common.all()
            and np.array_equal(
                merged.loc[common, "samuel_count"].to_numpy(),
                merged.loc[common, "adt_count"].to_numpy(),
            )
        ),
        "maximum_absolute_difference": float(
            np.max(
                np.abs(
                    merged.loc[common, "samuel_count"].to_numpy()
                    - merged.loc[common, "adt_count"].to_numpy()
                )
            )
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
