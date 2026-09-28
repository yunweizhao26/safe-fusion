#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr


PROTEIN_TO_GENE = {
    "CD86": "CD86",
    "PDL1": "CD274",
    "PDL2": "PDCD1LG2",
    "CD366": "HAVCR2",
}

PDL1_REGULATORS = {
    "BRD4": "up",
    "CUL3": "up",
    "IFNGR1": "down",
    "IFNGR2": "down",
    "IRF1": "down",
    "JAK2": "down",
    "STAT1": "down",
}


def checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)


def strip_lane(value: str) -> str:
    return re.sub(r"^l\d+_", "", str(value))


def lane(value: str) -> str:
    match = re.match(r"^(l\d+)_", str(value))
    return match.group(1) if match else ""


def read_geo_adt(raw_tar: Path) -> pd.DataFrame:
    barcode_name = "GSM4633615_ECCITE_ADT_Barcodes.csv.gz"
    count_name = "GSM4633615_ECCITE_ADT_counts.tsv.gz"
    with tarfile.open(raw_tar) as archive:
        barcode_member = archive.extractfile(barcode_name)
        count_member = archive.extractfile(count_name)
        if barcode_member is None or count_member is None:
            raise RuntimeError("ECCITE ADT files are missing from the GEO archive")
        with gzip.GzipFile(fileobj=barcode_member) as stream:
            mapping = pd.read_csv(
                io.BytesIO(stream.read()), header=None, names=["barcode", "protein"]
            )
        with gzip.GzipFile(fileobj=count_member) as stream:
            counts = pd.read_csv(io.BytesIO(stream.read()), sep="\t", index_col=0)
    counts.index = counts.index.astype(str).str.replace('"', "", regex=False)
    counts.columns = counts.columns.astype(str).str.replace('"', "", regex=False)
    protein_map = dict(zip(mapping["barcode"].astype(str), mapping["protein"].astype(str)))
    counts = counts.rename(index=protein_map)
    missing = sorted(set(PROTEIN_TO_GENE) - set(counts.index))
    if missing:
        raise RuntimeError(f"GEO ADT count rows are missing proteins: {missing}")
    return counts.loc[list(PROTEIN_TO_GENE)].T.astype(np.float64)


def find_column(frame: pd.DataFrame, choices: list[str]) -> str | None:
    for choice in choices:
        if choice in frame.columns:
            return choice
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mudata", required=True)
    parser.add_argument("--rna", required=True)
    parser.add_argument("--geo-raw-tar", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    mudata_path = Path(args.mudata)
    rna_path = Path(args.rna)
    raw_tar_path = Path(args.geo_raw_tar)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    multimodal = md.read_h5mu(mudata_path)
    adt = multimodal.mod["adt"]
    rna = ad.read_h5ad(rna_path)

    proteins = [str(value) for value in adt.var_names]
    if set(PROTEIN_TO_GENE) != set(proteins):
        raise RuntimeError(
            f"Expected proteins {sorted(PROTEIN_TO_GENE)}, observed {sorted(proteins)}"
        )
    adt_values = pd.DataFrame(
        as_dense(adt.X).astype(np.float64),
        index=adt.obs_names.astype(str),
        columns=proteins,
    )
    geo_values = read_geo_adt(raw_tar_path)
    common_cells = adt_values.index.intersection(geo_values.index)
    same_cell_set = set(adt_values.index) == set(geo_values.index)
    geo_equal = bool(
        same_cell_set
        and np.array_equal(
            adt_values.loc[geo_values.index, geo_values.columns].to_numpy(),
            geo_values.to_numpy(),
        )
    )
    max_geo_difference = float(
        np.max(
            np.abs(
                adt_values.loc[common_cells, geo_values.columns].to_numpy()
                - geo_values.loc[common_cells].to_numpy()
            )
        )
    )

    stripped = [strip_lane(value) for value in adt.obs_names]
    barcode_counts = collections.Counter(stripped)
    unambiguous = {key for key, value in barcode_counts.items() if value == 1}
    ambiguous = {key for key, value in barcode_counts.items() if value > 1}
    mudata_row = {
        strip_lane(cell_id): index
        for index, cell_id in enumerate(adt.obs_names.astype(str))
        if strip_lane(cell_id) in unambiguous
    }

    rna_index = {str(cell_id): index for index, cell_id in enumerate(rna.obs_names)}
    matched = [cell_id for cell_id in rna.obs_names.astype(str) if cell_id in mudata_row]
    guide_mudata_col = find_column(adt.obs, ["guide_ID", "guide_id"])
    guide_rna_col = find_column(rna.obs, ["guide_id", "guide_ID"])
    if guide_mudata_col is None or guide_rna_col is None:
        raise RuntimeError("Guide identifiers are required in both objects")
    guide_pairs = [
        (
            str(rna.obs.iloc[rna_index[cell_id]][guide_rna_col]),
            str(adt.obs.iloc[mudata_row[cell_id]][guide_mudata_col]),
        )
        for cell_id in matched
    ]
    guide_agreement = np.asarray([left == right for left, right in guide_pairs])
    if not bool(np.all(guide_agreement)):
        raise RuntimeError("The RNA and antibody guide labels do not agree exactly")

    target_col = find_column(adt.obs, ["gene_target", "target", "perturbation"])
    replicate_col = find_column(adt.obs, ["replicate", "orig.ident", "lane"])
    rows = []
    for cell_id in matched:
        mi = mudata_row[cell_id]
        mudata_cell_id = str(adt.obs_names[mi])
        target = str(adt.obs.iloc[mi][target_col]) if target_col else ""
        replicate = str(adt.obs.iloc[mi][replicate_col]) if replicate_col else lane(mudata_cell_id)
        for protein, gene in PROTEIN_TO_GENE.items():
            rows.append(
                {
                    "cell_id": cell_id,
                    "mudata_cell_id": mudata_cell_id,
                    "lane": lane(mudata_cell_id),
                    "replicate": replicate,
                    "guide_id": guide_pairs[len(rows) // len(PROTEIN_TO_GENE)][0],
                    "target": target,
                    "protein": protein,
                    "gene": gene,
                    "adt_count": float(adt_values.loc[mudata_cell_id, protein]),
                }
            )
    panel = pd.DataFrame(rows)
    wide = panel.pivot(index="cell_id", columns="protein", values="adt_count")
    clr = np.log1p(wide) - np.log1p(wide).mean(axis=1).to_numpy()[:, None]
    clr_long = clr.stack().rename("adt_clr").reset_index()
    panel = panel.merge(clr_long, on=["cell_id", "protein"], validate="one_to_one")
    panel.to_parquet(output / "matched_rna_adt_panel.parquet", index=False)
    panel.to_csv(output / "matched_rna_adt_panel.csv", index=False)

    rna_matrix = rna.layers["counts"] if "counts" in rna.layers else rna.X
    rna_gene_index = {str(gene): index for index, gene in enumerate(rna.var_names)}
    pair_rows = []
    for protein, gene in PROTEIN_TO_GENE.items():
        in_rna = gene in rna_gene_index
        record = {
            "protein": protein,
            "gene": gene,
            "in_full_rna": in_rna,
            "direct_target_present": bool(
                target_col and panel["target"].astype(str).eq(gene).any()
            ),
        }
        if in_rna:
            indices = np.asarray([rna_index[cell_id] for cell_id in matched], dtype=int)
            gene_values = rna_matrix[indices, rna_gene_index[gene]]
            values = as_dense(gene_values).reshape(-1).astype(float)
            protein_values = (
                panel.loc[panel["protein"] == protein]
                .set_index("cell_id")
                .loc[matched, "adt_clr"]
                .to_numpy(dtype=float)
            )
            rho, p_value = spearmanr(values, protein_values)
            record.update(
                {
                    "matched_cells": int(len(values)),
                    "rna_zero_cells": int(np.sum(values == 0)),
                    "rna_zero_fraction": float(np.mean(values == 0)),
                    "rna_protein_spearman": float(rho),
                    "rna_protein_spearman_p": float(p_value),
                }
            )
        pair_rows.append(record)
    pair_frame = pd.DataFrame(pair_rows)
    pair_frame.to_csv(output / "rna_protein_pair_audit.csv", index=False)

    pdl1 = panel.loc[panel["protein"] == "PDL1"].copy()
    target_medians = (
        pdl1.groupby("target", observed=True)["adt_count"]
        .agg(["size", "median", "mean"])
        .reset_index()
    )
    target_medians.to_csv(output / "pdl1_target_summary.csv", index=False)
    control_mask = pdl1["target"].astype(str).str.lower().str.contains(
        r"(^|[^a-z])(nt|ctrl|control|non.?target)", regex=True
    )
    if not bool(control_mask.any()):
        raise RuntimeError("Could not identify non-targeting controls")
    control_median = float(pdl1.loc[control_mask, "adt_count"].median())
    regulator_rows = []
    for regulator, expected in PDL1_REGULATORS.items():
        subset = pdl1.loc[pdl1["target"].astype(str) == regulator]
        observed = float(subset["adt_count"].median()) if len(subset) else np.nan
        direction = "up" if observed > control_median else "down" if observed < control_median else "equal"
        regulator_rows.append(
            {
                "regulator": regulator,
                "expected_direction": expected,
                "n_cells": int(len(subset)),
                "pdl1_median": observed,
                "control_median": control_median,
                "observed_direction": direction,
                "direction_agrees": bool(direction == expected),
            }
        )
    regulator_frame = pd.DataFrame(regulator_rows)
    regulator_frame.to_csv(output / "pdl1_regulator_direction_check.csv", index=False)

    report = {
        "files": {
            "mudata": {"path": str(mudata_path), "md5": checksum(mudata_path, "md5")},
            "rna": {"path": str(rna_path), "sha256": checksum(rna_path, "sha256")},
            "geo_raw_tar": {"path": str(raw_tar_path), "sha256": checksum(raw_tar_path, "sha256")},
        },
        "modalities": sorted(multimodal.mod.keys()),
        "proteins": proteins,
        "geo_comparison": {
            "mudata_cells": int(len(adt_values)),
            "geo_cells": int(len(geo_values)),
            "common_cells": int(len(common_cells)),
            "same_cell_set": same_cell_set,
            "exact_count_match": geo_equal,
            "maximum_absolute_difference": max_geo_difference,
        },
        "barcode_join": {
            "lane_prefixed_cells": int(len(stripped)),
            "distinct_stripped_barcodes": int(len(barcode_counts)),
            "ambiguous_stripped_barcodes_dropped": int(len(ambiguous)),
            "physical_cells_dropped_for_ambiguity": int(
                sum(barcode_counts[value] for value in ambiguous)
            ),
            "unambiguous_rna_matches": int(len(matched)),
            "guide_agreement_count": int(guide_agreement.sum()),
            "guide_agreement_fraction": float(guide_agreement.mean()),
        },
        "pdl1_regulator_check": {
            "control_cells": int(control_mask.sum()),
            "control_median": control_median,
            "directions_agree": int(regulator_frame["direction_agrees"].sum()),
            "directions_tested": int(len(regulator_frame)),
        },
        "interpretation": {
            "independent_primary_pair": "PDL1 protein and CD274 RNA",
            "additional_non_targeted_pair": "CD366 protein and HAVCR2 RNA",
            "direct_target_sensitivity_pairs": ["CD86 protein and CD86 RNA", "PDL2 protein and PDCD1LG2 RNA"],
            "assumption": "Within RNA-zero cells, higher matched protein abundance supports residual expression but does not establish the biological cause of an individual RNA zero.",
        },
    }
    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
