#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pertpy as pt
import scanpy as sc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mudata", default="external_data/papalexi_multimodal/papalexi.h5mu")
    parser.add_argument("--prepared", default="external_data/prepared/papalexi_eccite.h5ad")
    parser.add_argument("--output-dir", default="artifacts/paper_evidence/review_round2/knockdown/mixscape")
    parser.add_argument("--seed", type=int, default=0, help="Mixscape random_state (the pertpy default).")
    args = parser.parse_args()

    rna = md.read_h5mu(args.mudata)["rna"].copy()
    rna.layers["counts"] = rna.X.copy()
    sc.pp.normalize_total(rna)
    sc.pp.log1p(rna)
    sc.pp.highly_variable_genes(rna, subset=True)
    sc.pp.pca(rna, random_state=args.seed)
    mixscape = pt.tl.Mixscape()
    mixscape.perturbation_signature(rna, "perturbation", "NT", split_by="replicate")
    mixscape.mixscape(adata=rna, pert_key="gene_target", control="NT", layer="X_pert", random_state=args.seed)

    classes = rna.obs[["gene_target", "guide_ID", "MULTI_ID", "mixscape_class", "mixscape_class_global", "mixscape_class_p_ko"]].copy()
    classes["barcode"] = [name.split("_", 1)[1] for name in classes.index.astype(str)]
    classes["mudata_cell_id"] = classes.index.astype(str)
    key = ["barcode", "MULTI_ID", "guide_ID"]
    unique = classes[~classes.duplicated(key, keep=False)]

    prepared = ad.read_h5ad(args.prepared, backed="r").obs
    cells = pd.DataFrame({
        "cell_id": prepared.index.astype(str),
        "barcode": prepared["source_cell_id"].astype(str).to_numpy(),
        "MULTI_ID": prepared["hto"].astype(str).to_numpy(),
        "guide_ID": prepared["guide_id"].astype(str).to_numpy(),
        "target": prepared["target"].astype(str).to_numpy(),
    })
    joined = cells.merge(unique, on=key, how="left", validate="one_to_one")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output / "papalexi_mixscape_classes.parquet", index=False)
    classes.reset_index(drop=True).to_parquet(output / "papalexi_mixscape_all_cells.parquet", index=False)
    matched = joined["mixscape_class_global"].notna()
    target_agrees = (joined.loc[matched, "target"].replace("none", "NT") == joined.loc[matched, "gene_target"].astype(str)).all()
    summary = {
        "versions": {name: version(name) for name in ("pertpy", "scanpy", "mudata")},
        "mudata_cells": int(len(classes)),
        "prepared_cells": int(len(cells)),
        "prepared_cells_matched": int(matched.sum()),
        "matched_targets_agree": bool(target_agrees),
        "global_class_counts_all_cells": classes["mixscape_class_global"].astype(str).value_counts().to_dict(),
        "global_class_counts_prepared_cells": joined["mixscape_class_global"].astype(str).value_counts().to_dict(),
        "knockout_share_by_target_all_cells": (
            classes[classes["gene_target"].astype(str) != "NT"]
            .groupby(classes["gene_target"].astype(str))["mixscape_class_global"]
            .apply(lambda values: float(np.mean(values.astype(str) == "KO")))
            .round(4).to_dict()
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
