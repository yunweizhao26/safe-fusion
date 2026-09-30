#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from sle_common import CASE, FOLDS, METHODS, REPORTED, SEED, bootstrap_draws, config, filled_matrix, interval, load_fills, load_unit

TREG = "T4_reg"

def log_normalize(counts: np.ndarray) -> np.ndarray:
    library = counts.sum(axis=1, keepdims=True)
    return np.log1p(1e4 * np.divide(counts, library, out=np.zeros_like(counts), where=library > 0))

def cluster(counts: np.ndarray, components: int, resolution: float, seed: int) -> np.ndarray:
    adata = ad.AnnData(X=log_normalize(counts).astype(np.float32))
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=components, random_state=0)
    sc.pp.neighbors(adata, n_neighbors=15, random_state=0)
    sc.tl.leiden(adata, resolution=resolution, random_state=seed, flavor="igraph", n_iterations=2, directed=False)
    return adata.obs["leiden"].astype(int).to_numpy()

def best_match(labels: np.ndarray, target: np.ndarray) -> dict:
    best = {"f1": -1.0}
    for value in np.unique(labels):
        members = labels == value
        true_positive = float((members & target).sum())
        f1 = 2 * true_positive / (members.sum() + target.sum())
        if f1 > best["f1"]:
            best = {"cluster": int(value), "size": int(members.sum()), "f1": f1,
                    "precision": true_positive / members.sum(), "recall": true_positive / target.sum()}
    return best

def high_subclusters(labels: np.ndarray, detected: np.ndarray) -> dict:
    rows, p_values = [], []
    for value in np.unique(labels):
        members = labels == value
        table = [[int((detected & members).sum()), int((~detected & members).sum())],
                 [int((detected & ~members).sum()), int((~detected & ~members).sum())]]
        p_values.append(stats.fisher_exact(table, alternative="greater").pvalue)
        rows.append({"size": int(members.sum()), "detection": float(detected[members].mean())})
    q = stats.false_discovery_control(np.asarray(p_values))
    high = [row for row, value in zip(rows, q) if value <= 0.05]
    return {"subclusters": len(rows), "high_subclusters": len(high), "cells_in_high_subclusters": int(sum(row["size"] for row in high)),
            "max_detection": max(row["detection"] for row in rows), "treg_detection": float(detected.mean())}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", dest="set_name", default="main")
    parser.add_argument("--output-dir", type=Path, default=CASE / "results" / "q3_clustering")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()
    focus = config()["focus"]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    cluster_rows, call_rows, sub_rows = [], [], []
    call_counts = []
    for fold in range(FOLDS):
        unit = load_unit(args.set_name, fold, "deploy")
        test_rows = np.flatnonzero(unit.test)
        obs = unit.obs.iloc[test_rows]
        author_treg = (obs["cell_type"] == TREG).to_numpy()
        fills = {method: load_fills(unit, method) for method in METHODS}
        raw = unit.counts[test_rows]
        variants = [("Unfilled", "none", 0, raw)] + [
            (method, mode, level, filled_matrix(unit, fills[method], level, mode))
            for mode in ("global", "gene") for method in METHODS for level in REPORTED
        ]

        training = ~unit.test
        scaler = StandardScaler().fit(log_normalize(unit.recorded[training]))
        classifier = LogisticRegression(max_iter=2000, C=1.0)
        classifier.fit(scaler.transform(log_normalize(unit.recorded[training])), unit.obs["cell_type"].to_numpy()[training])

        base_labels = cluster(raw, 30, 1.0, 0)
        reference_labels = cluster(raw, 30, 1.0, 1)
        base_calls = classifier.predict(scaler.transform(log_normalize(raw)))
        treg_counts = raw[author_treg]
        sub_base = cluster(treg_counts, 20, 0.5, 0)
        sub_reference = cluster(treg_counts, 20, 0.5, 1)
        cluster_rows.append({"fold": fold, "method": "Unfilled, second Leiden seed", "budget": "none", "fill_pct": 0,
                             "ari": adjusted_rand_score(base_labels, reference_labels),
                             "nmi": normalized_mutual_info_score(base_labels, reference_labels),
                             "treg_subcluster_ari": adjusted_rand_score(sub_base, sub_reference),
                             **{f"treg_cluster_{key}": value for key, value in best_match(reference_labels, author_treg).items()}})
        for method, mode, level, matrix in variants:
            labels = base_labels if method == "Unfilled" else cluster(matrix, 30, 1.0, 0)
            subs = sub_base if method == "Unfilled" else cluster(matrix[author_treg], 20, 0.5, 0)
            cluster_rows.append({"fold": fold, "method": method, "budget": mode, "fill_pct": level,
                                 "ari": adjusted_rand_score(base_labels, labels), "nmi": normalized_mutual_info_score(base_labels, labels),
                                 "treg_subcluster_ari": adjusted_rand_score(sub_base, subs),
                                 **{f"treg_cluster_{key}": value for key, value in best_match(labels, author_treg).items()}})
            calls = base_calls if method == "Unfilled" else classifier.predict(scaler.transform(log_normalize(matrix)))
            before, after = base_calls == TREG, calls == TREG
            frame = pd.DataFrame({"donor": obs["donor"].to_numpy(), "condition": obs["condition"].to_numpy(),
                                  "before": before, "after": after, "into": ~before & after, "out": before & ~after,
                                  "author_treg": author_treg})
            per_donor = frame.groupby(["donor", "condition"])[["before", "after", "into", "out", "author_treg"]].sum().reset_index()
            per_donor["called_and_author"] = frame.assign(both=after & author_treg).groupby("donor")["both"].sum().reindex(per_donor["donor"]).to_numpy()
            per_donor.insert(0, "fill_pct", level)
            per_donor.insert(0, "budget", mode)
            per_donor.insert(0, "method", method)
            per_donor.insert(0, "fold", fold)
            call_counts.append(per_donor)
            for gene in focus:
                g = unit.genes.index(gene)
                sub_rows.append({"fold": fold, "method": method, "budget": mode, "fill_pct": level, "gene": gene,
                                 **high_subclusters(subs, matrix[author_treg, g] > 0)})
            print(f"fold {fold} {method} {mode} {level}", flush=True)

    clusters = pd.DataFrame(cluster_rows)
    counts = pd.concat(call_counts, ignore_index=True)
    subclusters = pd.DataFrame(sub_rows)
    clusters.to_csv(output / "clustering_by_fold.csv", index=False)
    counts.to_csv(output / "treg_calls_by_donor.csv", index=False)
    subclusters.to_csv(output / "treg_subclusters_by_fold.csv", index=False)

    donors = counts.loc[counts["method"] == "Unfilled"].drop_duplicates("donor").set_index("donor")["condition"].sort_index()
    weights = np.vstack([np.ones(len(donors)), bootstrap_draws(donors.index.to_numpy(), donors.to_numpy(), args.draws, SEED)])
    base = counts.loc[counts["method"] == "Unfilled"].set_index("donor").reindex(donors.index)
    summary_rows = []
    for (method, mode, level), frame in counts.groupby(["method", "budget", "fill_pct"], sort=False):
        frame = frame.set_index("donor").reindex(donors.index)
        row = {"method": method, "budget": mode, "fill_pct": level}
        for cond in ("all", "SLE", "healthy"):
            members = np.ones(len(donors), bool) if cond == "all" else (donors.to_numpy() == cond)
            w = weights * members
            before = w @ base["before"].to_numpy(float)
            after = w @ frame["after"].to_numpy(float)
            change = (after - before) / before
            row[f"treg_calls_before_{cond}"] = int(before[0])
            row[f"treg_calls_after_{cond}"] = int(after[0])
            row[f"relative_change_{cond}"] = float(change[0])
            row[f"relative_change_{cond}_ci"] = interval(change[1:])
            row[f"into_treg_{cond}"] = int(frame.loc[members, "into"].sum())
            row[f"out_of_treg_{cond}"] = int(frame.loc[members, "out"].sum())
        row["author_tregs"] = int(frame["author_treg"].sum())
        row["called_treg_precision"] = float(frame["called_and_author"].sum() / max(frame["after"].sum(), 1))
        row["called_treg_recall"] = float(frame["called_and_author"].sum() / frame["author_treg"].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "treg_calls_summary.csv", index=False)
    pooled = clusters.groupby(["method", "budget", "fill_pct"], sort=False)[["ari", "nmi", "treg_subcluster_ari", "treg_cluster_size", "treg_cluster_f1"]].mean().reset_index()
    pooled.to_csv(output / "clustering_mean_over_folds.csv", index=False)
    print(pooled.to_string(index=False))
    print(summary.to_string(index=False))
    subcluster_summary = subclusters.groupby(["method", "budget", "fill_pct", "gene"], sort=False).agg(
        high_subclusters=("high_subclusters", "sum"), cells_in_high_subclusters=("cells_in_high_subclusters", "sum"),
        max_detection_mean=("max_detection", "mean"), treg_detection_mean=("treg_detection", "mean")).reset_index()
    subcluster_summary.to_csv(output / "treg_subclusters_summary.csv", index=False)
    print(subcluster_summary.to_string(index=False))

if __name__ == "__main__":
    main()
