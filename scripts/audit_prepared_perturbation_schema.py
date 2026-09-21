#!/usr/bin/env python3
"""Write compact observation-schema summaries for prepared perturbation objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, help="name=path")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reports = {}
    for value in args.input:
        name, path = value.split("=", 1)
        adata = ad.read_h5ad(path, backed="r")
        columns = {}
        for column in adata.obs.columns:
            series = adata.obs[column].astype(str)
            counts = series.value_counts().head(20)
            columns[column] = {
                "n_unique": int(series.nunique()),
                "top_values": {str(key): int(count) for key, count in counts.items()},
            }
        reports[name] = {
            "cells": int(adata.n_obs),
            "genes": int(adata.n_vars),
            "obs_columns": columns,
        }
        adata.file.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
