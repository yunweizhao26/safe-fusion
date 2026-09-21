#!/usr/bin/env python3
"""Reproduce the Crohn disagreement figures from supplied analysis inputs.

Matrix rows are streamed to limit memory use.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATASET = "63ff2c52-cb63-44f0-bac3-d0b33373e312"
METHODS = ("SAUCIE", "MAGIC", "deepImpute", "scScope", "scVI", "knn_smoothing")
SUBSETS = (
    ("Crohn disease", "caecum epithelium"),
    ("Crohn disease", "caecum"),
    ("Crohn disease", "colon"),
    ("Crohn disease", "colonic epithelium"),
    ("Crohn disease", "lamina propria of mucosa of colon"),
    ("Crohn disease", "left colon"),
    ("Crohn disease", "right colon"),
    ("Crohn disease", "sigmoid colon"),
    ("normal", "colon"),
)
MARKER_GENES = {
    "Enterocytes BEST4": ["BEST4", "OTOP2", "CA7", "GUCA2A", "GUCA2B", "SPIB", "CFTR"],
    "Goblet cells MUC2 TFF1": ["MUC2", "TFF1", "TFF3", "FCGBP", "AGR2", "SPDEF"],
    "Tuft cells": ["POU2F3", "DCLK1"],
    "Goblet cells SPINK4": ["MUC2", "SPINK4"],
    "Enterocytes TMIGD1 MEP1A": ["CA1", "CA2", "TMIGD1", "MEP1A"],
    "Enterocytes CA1 CA2 CA4-": ["CA1", "CA2"],
    "Goblet cells MUC2 TFF1-": ["MUC2"],
    "Epithelial Cycling cells": ["LGR5", "OLFM4", "MKI67"],
    "Enteroendocrine cells": ["CHGA", "GCG", "GIP", "CCK"],
    "Stem cells OLFM4": ["OLFM4", "LGR5"],
    "Stem cells OLFM4 LGR5": ["OLFM4", "LGR5", "ASCL2"],
    "Stem cells OLFM4 PCNA": ["OLFM4", "PCNA", "LGR5", "ASCL2", "SOX9", "TERT"],
    "Paneth cells": ["LYZ", "DEFA5"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", "--legacy-root", dest="legacy_root", type=Path, required=True,
        help="Directory containing the source ep_dataset/ and output/ trees.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--only", choices=("all", "marker", "heatmap"), default="all")
    parser.add_argument(
        "--candidate-source",
        choices=("observed-h5ad", "legacy"),
        default="observed-h5ad",
        help="Use observed zeros from the source h5ad or reproduce the mixed legacy protocol.",
    )
    return parser.parse_args()


def reproduce_marker_panel(legacy_root: Path, output_dir: Path) -> None:
    source = legacy_root / "output" / "analysis" / "include_random_2.json"
    with source.open() as handle:
        data = json.load(handle)

    reference_genes = {gene for genes in MARKER_GENES.values() for gene in genes}
    distributions: list[tuple[str, list[float]]] = []
    table_rows: list[dict[str, float | str]] = []

    for dataset_name, dataset_values in data.items():
        pair = dataset_values.get("SAUCIE", {}).get("MAGIC", {})
        per_gene = pair.get("per_gene_concordance", [])
        values = [value for gene, value in per_gene if gene in reference_genes]
        if values:
            distributions.append((dataset_name, sorted(values, reverse=True)))

        actual = pair.get("contingency", {})
        random = pair.get("contingency_random", {})
        if not actual or not random:
            continue

        def agreement(contingency: dict[str, int]) -> float:
            total = sum(
                contingency[key]
                for key in (
                    "both_imputed",
                    "only_method_a_imputed",
                    "only_method_b_imputed",
                    "neither_imputed",
                )
            )
            return (
                contingency["both_imputed"] + contingency["neither_imputed"]
            ) / total

        table_rows.append(
            {
                "dataset_name": dataset_name,
                "overall_conc_actual": agreement(actual),
                "overall_conc_random": agreement(random),
            }
        )

    ncols = math.ceil(math.sqrt(len(distributions)))
    nrows = math.ceil(len(distributions) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for axis, (dataset_name, values) in zip(flat_axes, distributions):
        sns.histplot(
            values,
            color="blue",
            bins=30,
            element="step",
            fill=False,
            stat="count",
            ax=axis,
        )
        axis.set_title(dataset_name, fontsize=10)
        axis.set_xlabel("Concordance")
        axis.set_ylabel("Ref Genes")
    for axis in flat_axes[len(distributions) :]:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "all_gene_distributions_ref_genes.png")
    plt.close(fig)

    pd.DataFrame(table_rows).to_csv(output_dir / "overall_concordance_table.csv", index=False)
    print(
        f"Marker panel uses SAUCIE versus MAGIC and "
        f"{len(reference_genes)} unique marker genes."
    )


def load_method_arrays(
    legacy_root: Path, disease: str, tissue: str
) -> tuple[list[np.ndarray], tuple[int, int]]:
    arrays: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None
    for method in METHODS:
        path = legacy_root / "output" / method / DATASET / disease / f"{tissue}.npy"
        array = np.load(path, mmap_mode="r")
        if target_shape is None:
            target_shape = array.shape
        if array.shape[0] != target_shape[0] or array.shape[1] > target_shape[1]:
            raise ValueError(f"Unsupported shape for {method}: {array.shape} versus {target_shape}")
        arrays.append(array)
    assert target_shape is not None
    return arrays, target_shape


def padded_chunk(
    array: np.ndarray,
    start: int,
    stop: int,
    n_genes: int,
    gene_indices: np.ndarray | None = None,
) -> np.ndarray:
    chunk = np.asarray(array[start:stop])
    if chunk.shape[1] == n_genes:
        return chunk
    padded = np.zeros((stop - start, n_genes), dtype=chunk.dtype)
    if gene_indices is None:
        padded[:, : chunk.shape[1]] = chunk
    else:
        if len(gene_indices) != chunk.shape[1]:
            raise ValueError(
                f"MAGIC has {chunk.shape[1]} columns but {len(gene_indices)} genes passed filtering"
            )
        padded[:, gene_indices] = chunk
    return padded


class H5adObservedMatrix:
    def __init__(
        self,
        legacy_root: Path,
        disease: str,
        tissue: str,
        expected_shape: tuple[int, int],
        scan_rows: int = 512,
    ) -> None:
        import anndata as ad

        source = legacy_root / "datasets" / f"{DATASET}.h5ad"
        self.adata = ad.read_h5ad(source, backed="r")
        obs = self.adata.obs
        mask = (obs["disease"].to_numpy() == disease) & (obs["tissue"].to_numpy() == tissue)
        self.indices = np.flatnonzero(mask)
        self.alignment = "all matching source cells"
        if (len(self.indices), self.adata.n_vars) != expected_shape:
            # The archived normal colon outputs were generated only for rows
            # with recorded age group metadata.  Keep that historical row
            # selection explicit instead of truncating the source matrix.
            if "age group" in obs.columns:
                has_age = obs["age group"].notna().to_numpy()
                age_aligned = np.flatnonzero(mask & has_age)
                if (len(age_aligned), self.adata.n_vars) == expected_shape:
                    self.indices = age_aligned
                    self.alignment = "matching source cells with recorded age group"
        if (len(self.indices), self.adata.n_vars) != expected_shape:
            raise ValueError(
                f"Observed subset shape {(len(self.indices), self.adata.n_vars)} "
                f"does not match method shape {expected_shape}"
            )

        nonzero_counts = np.zeros(self.adata.n_vars, dtype=np.int64)
        for start in range(0, len(self.indices), scan_rows):
            stop = min(start + scan_rows, len(self.indices))
            block = self.adata.X[self.indices[start:stop]]
            if hasattr(block, "getnnz"):
                nonzero_counts += np.asarray(block.getnnz(axis=0)).ravel()
            else:
                nonzero_counts += np.count_nonzero(np.asarray(block), axis=0)
        self.magic_gene_indices = np.flatnonzero(nonzero_counts >= 5)

    def chunk(self, start: int, stop: int) -> np.ndarray:
        block = self.adata.X[self.indices[start:stop]]
        return block.toarray() if hasattr(block, "toarray") else np.asarray(block)

    def close(self) -> None:
        if self.adata.file is not None:
            self.adata.file.close()


def disagreement_for_subset(
    legacy_root: Path,
    disease: str,
    tissue: str,
    chunk_rows: int,
    candidate_source: str,
) -> tuple[pd.DataFrame, str, int]:
    arrays, (n_cells, n_genes) = load_method_arrays(legacy_root, disease, tissue)
    observed_reader: H5adObservedMatrix | None = None
    if candidate_source == "observed-h5ad":
        observed_reader = H5adObservedMatrix(
            legacy_root, disease, tissue, (n_cells, n_genes)
        )
        original = None
        protocol = f"observed zeros from source h5ad; {observed_reader.alignment}"
        magic_gene_indices = observed_reader.magic_gene_indices
    else:
        original_path = legacy_root / "output" / "original" / DATASET / disease / f"{tissue}.npy"
        original = np.load(original_path, mmap_mode="r") if original_path.exists() else None
        protocol = "observed zeros" if original is not None else "legacy low value proxy"
        magic_gene_indices = None

    pattern_counts = np.zeros(2 ** len(METHODS), dtype=np.int64)
    try:
        for start in range(0, n_cells, chunk_rows):
            stop = min(start + chunk_rows, n_cells)
            decision_code = np.zeros((stop - start, n_genes), dtype=np.uint8)
            if observed_reader is not None:
                candidate = observed_reader.chunk(start, stop) == 0
                threshold = 0.0
            elif original is not None:
                candidate = np.asarray(original[start:stop]) == 0
                threshold = 0.0
            else:
                low_count = np.zeros((stop - start, n_genes), dtype=np.uint8)
                threshold = 0.01

            for method_index, (method, array) in enumerate(zip(METHODS, arrays)):
                indices = magic_gene_indices if method == "MAGIC" else None
                chunk = padded_chunk(array, start, stop, n_genes, indices)
                if observed_reader is None and original is None:
                    low_count += chunk <= 0.01
                decision_code |= (chunk > threshold).astype(np.uint8) << method_index

            if observed_reader is None and original is None:
                candidate = low_count >= len(METHODS) // 2

            pattern_counts += np.bincount(
                decision_code[candidate], minlength=len(pattern_counts)
            )
    finally:
        if observed_reader is not None:
            observed_reader.close()

    denominator = int(pattern_counts.sum())
    if denominator == 0:
        raise ValueError(f"No candidate entries for {disease}, {tissue}")
    pair_counts = np.zeros((len(METHODS), len(METHODS)), dtype=np.int64)
    patterns = np.arange(len(pattern_counts), dtype=np.uint8)
    for left in range(len(METHODS)):
        for right in range(left + 1, len(METHODS)):
            disagree = ((patterns >> left) & 1) != ((patterns >> right) & 1)
            count = int(pattern_counts[disagree].sum())
            pair_counts[left, right] = count
            pair_counts[right, left] = count
    rates = pair_counts / denominator
    return pd.DataFrame(rates, index=METHODS, columns=METHODS), protocol, denominator


def reproduce_heatmap(
    legacy_root: Path, output_dir: Path, chunk_rows: int, candidate_source: str
) -> None:
    disagreement_data: list[tuple[str, pd.DataFrame]] = []
    protocol_rows: list[dict[str, str | int]] = []
    matrix_dir = output_dir / "disagreement_matrices"
    matrix_dir.mkdir(exist_ok=True)

    for disease, tissue in SUBSETS:
        subset_name = f"{disease} - {tissue}"
        frame, protocol, denominator = disagreement_for_subset(
            legacy_root, disease, tissue, chunk_rows, candidate_source
        )
        disagreement_data.append((subset_name, frame))
        slug = subset_name.replace(" - ", "_").replace(" ", "_").lower()
        frame.to_csv(matrix_dir / f"{slug}.csv")
        protocol_rows.append(
            {"subset": subset_name, "candidate_protocol": protocol, "candidate_entries": denominator}
        )
        values = frame.to_numpy()[np.triu_indices(len(METHODS), k=1)]
        print(f"{subset_name}: mean disagreement {values.mean():.6f} using {protocol}")

    pd.DataFrame(protocol_rows).to_csv(output_dir / "disagreement_protocols.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle("Pairwise Disagreement Between Methods", fontsize=18, y=0.98)
    for index, (subset_name, frame) in enumerate(disagreement_data):
        axis = axes.flatten()[index]
        plot_frame = frame.rename(
            index={"knn_smoothing": "KNN Smoothing"},
            columns={"knn_smoothing": "KNN Smoothing"},
        )
        sns.heatmap(
            plot_frame,
            annot=True,
            fmt=".3f",
            cmap="OrRd",
            square=True,
            annot_kws={"fontsize": 11},
            cbar_kws={"shrink": 0.8},
            ax=axis,
        )
        axis.set_title(subset_name, fontsize=13, pad=10)
        axis.tick_params(axis="x", rotation=45, labelsize=11)
        axis.tick_params(axis="y", rotation=0, labelsize=11)
        colorbar = axis.collections[0].colorbar
        if colorbar:
            colorbar.ax.tick_params(labelsize=11)
            if index == len(disagreement_data) - 1:
                colorbar.set_label("Disagreement Rate", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    fig.savefig(output_dir / "combined_disagreement_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.only in ("all", "marker"):
        reproduce_marker_panel(args.legacy_root, args.output_dir)
    if args.only in ("all", "heatmap"):
        reproduce_heatmap(
            args.legacy_root, args.output_dir, args.chunk_rows, args.candidate_source
        )


if __name__ == "__main__":
    main()
