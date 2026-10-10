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
from sklearn.metrics import roc_auc_score

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "scripts"))

EVIDENCE = REPOSITORY / "artifacts/paper_evidence"
KNOCKDOWN_SCREENS = ("adamson_crispri", "papalexi_eccite", "norman_crispra")
SEX_TISSUES = {"pancreas": ("pancreas_0", "pancreas_1", "pancreas_2"), "colon": ("colon",)}

def dense(value) -> np.ndarray:
    return value.toarray() if sparse.issparse(value) else np.asarray(value)

def candidate_scores(input_path: Path, splits_path: Path, scores_path: Path) -> tuple[np.ndarray, np.ndarray, int]:

    adata = ad.read_h5ad(input_path)
    counts = dense(adata.layers["corrupted_counts"])
    split = pd.read_parquet(splits_path).set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    rows, cols = np.where((counts == 0) & (split == "test")[:, None])
    scores = np.load(scores_path).astype(np.float64)
    if len(scores) != len(rows):
        raise ValueError(f"{scores_path}: {len(scores)} scores for {len(rows)} held-out zeros")
    matrix = np.full(counts.shape, np.nan)
    matrix[rows, cols] = scores
    order = np.argsort(-scores, kind="stable")
    rank = np.full(counts.shape, -1, dtype=np.int64)
    rank[rows[order], cols[order]] = np.arange(len(order))
    return matrix, rank, len(rows)

def first_fill(rank: np.ndarray, n: int, fractions) -> np.ndarray:

    order = np.full(rank.shape, np.inf, dtype=np.float32)
    for pct in sorted(fractions, reverse=True):
        order[(rank >= 0) & (rank < max(1, int(round(pct / 100 * n))))] = pct
    return order

def knockdown(args: argparse.Namespace, roots: dict[str, Path], output: Path) -> None:
    import evaluate_knockdown_zero_analyses as kd
    from evaluate_perturbation_zeros import FRACTIONS, eligible_targets, likely_dropout_labels, load_screen

    defaults = argparse.Namespace(
        deploy_root="artifacts/paper_evidence/downstream_deployment",
        review_root="artifacts/paper_evidence/review_round2/knockdown",
        norman_benchmark="artifacts/paper_evidence/review_round2/leakage_free/norman_crispra",
        norman_root="artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt")
    rows, fill_rows = [], []
    for dataset in KNOCKDOWN_SCREENS:
        inputs = kd.screen_inputs(dataset, defaults)
        screen = load_screen(dataset, inputs.deploy, inputs.splits, inputs.prepared)
        targets = eligible_targets(screen, args.min_detection)
        log_library = np.log(screen.recorded.sum(axis=1, dtype=np.float64))
        for name, root in roots.items():
            score, rank, n = candidate_scores(inputs.deploy / "hybrid.h5ad", inputs.deploy / "splits.parquet",
                                              root / dataset / "test_scores.npy")
            order = first_fill(rank, n, FRACTIONS)
            for item in targets:
                labels = likely_dropout_labels(screen, item)
                cells = item.cells
                values = {"fill_order": np.where(np.isfinite(order[cells, item.g]), 11 - order[cells, item.g], 0.0),
                          "continuous": score[cells, item.g]}
                if np.isnan(values["continuous"]).any():
                    raise ValueError(f"{dataset} {item.gene}: a target zero has no score")
                for score_type, value in values.items():
                    strata, _ = kd.stratified_auroc(labels, value, log_library[cells], args.depth_strata)
                    rows.append({"dataset": dataset, "direction": screen.direction, "gene": item.gene, "method": name,
                                 "score_type": score_type, "none": kd.auroc(labels, value), "depth_strata": strata})
                for pct in (1, 5, 10):
                    filled = order[cells, item.g] <= pct
                    fill_rows.append({"dataset": dataset, "gene": item.gene, "method": name, "fill_pct": pct,
                                      "fill_likely_dropout": float(filled[labels == 1].mean()),
                                      "fill_likely_biological": float(filled[labels == 0].mean())})
    table = pd.DataFrame(rows)
    fills = pd.DataFrame(fill_rows)
    summary = []
    for dataset in KNOCKDOWN_SCREENS:
        genes = table.loc[table.dataset == dataset, "gene"].unique()
        draws = np.random.default_rng([args.seed, kd.SCOPES.index(dataset)]).integers(0, len(genes), size=(args.draws, len(genes)))
        for score_type in ("fill_order", "continuous"):
            frame = table[(table.dataset == dataset) & (table.score_type == score_type)]
            wide = {adjustment: frame.pivot(index="gene", columns="method", values=adjustment).loc[genes]
                    for adjustment in ("none", "depth_strata")}
            for name in roots:
                for adjustment, values in wide.items():
                    summary.append({"dataset": dataset, "n_targets": len(genes), "score_type": score_type,
                                    "adjustment": adjustment, "method": name, "quantity": "auroc",
                                    **dict(zip(("estimate", "low", "high"), kd.interval(values[name].to_numpy(), draws)))})
                    if name != args.reference:
                        difference = (values[name] - values[args.reference]).to_numpy()
                        summary.append({"dataset": dataset, "n_targets": len(genes), "score_type": score_type,
                                        "adjustment": adjustment, "method": name,
                                        "quantity": f"minus {args.reference}",
                                        **dict(zip(("estimate", "low", "high"), kd.interval(difference, draws)))})
        for name in roots:
            for pct in (1, 5, 10):
                frame = fills[(fills.dataset == dataset) & (fills.method == name) & (fills.fill_pct == pct)].set_index("gene").loc[genes]
                for column in ("fill_likely_dropout", "fill_likely_biological"):
                    summary.append({"dataset": dataset, "n_targets": len(genes), "score_type": f"fill_{pct}pct",
                                    "adjustment": column, "method": name, "quantity": "fill rate",
                                    **dict(zip(("estimate", "low", "high"), kd.interval(frame[column].to_numpy(), draws)))})
    table.to_csv(output / "knockdown_per_target.csv", index=False, float_format="%.5f")
    pd.DataFrame(summary).to_csv(output / "knockdown_summary.csv", index=False, float_format="%.5f")
    print(pd.DataFrame(summary).round(3).to_string(index=False))

def sex(args: argparse.Namespace, roots: dict[str, Path], output: Path) -> None:
    from disease_control_common import DEPLOYMENT, OUTPUT, load_heldout, stratified_draws
    from disease_control_sex_zeros import SEX_GENES, summarize, zero_table

    fractions = (0.01, 0.05, 0.10)
    auroc_frames, fill_frames, paired = [], [], []
    for tissue, keys in SEX_TISSUES.items():
        data = load_heldout(tissue)
        donor_sex = pd.read_csv(OUTPUT / "sex_zeros" / tissue / "donor_sex.csv").set_index("donor")["sex"]
        data.obs["sex"] = data.obs["donor"].map(donor_sex)
        symbols = list(data.symbols)
        gene_index = {gene: symbols.index(gene) for gene in SEX_GENES}
        cells = data.cells
        donors = sorted(cells["donor"].unique())
        draws = stratified_draws(cells.groupby("donor")["sex"].first().loc[donors].to_numpy(), args.draws, args.seed)
        table = zero_table(cells, data.counts, gene_index, None)
        full_rows = np.flatnonzero(data.heldout)[table["row"].to_numpy()]
        cols = table["col"].to_numpy()
        for name, root in roots.items():
            score = np.full(len(table), np.nan)
            filled = {fraction: np.zeros(len(table), dtype=bool) for fraction in fractions}
            for key in keys:
                deploy = DEPLOYMENT / "pancreas" / f"fold_{key.split('_')[1]}" if key.startswith("pancreas") else DEPLOYMENT / key
                matrix, rank, n = candidate_scores(deploy / "hybrid.h5ad", deploy / "splits.parquet", root / key / "test_scores.npy")
                mine = data.test_masks[key][full_rows]
                score[mine] = matrix[full_rows[mine], cols[mine]]
                for fraction in fractions:
                    selected = rank[full_rows[mine], cols[mine]]
                    filled[fraction][mine] = (selected >= 0) & (selected < max(1, int(round(fraction * n))))
            if np.isnan(score).any():
                raise ValueError(f"{tissue} {name}: a sex-linked zero has no score")
            table[f"score:{name}"] = score
            for fraction in fractions:
                table[f"filled:{name}:{fraction}"] = filled[fraction]
        fill_rows, auroc_rows = summarize(table, list(roots), ["recorded zero, expressing sex"], draws, donors, "recorded counts")
        auroc_frames.append(pd.DataFrame(auroc_rows).assign(tissue=tissue))
        fill_frames.append(pd.DataFrame(fill_rows).assign(tissue=tissue))
        donor_code = pd.Categorical(table["donor"], categories=donors).codes
        weights = np.stack([np.bincount(draw, minlength=len(donors))[donor_code] for draw in draws]).astype(float)
        for gene in SEX_GENES:
            keep = (table["gene"] == gene).to_numpy() & table["kind"].isin(["recorded zero, expressing sex", "absent, other sex"]).to_numpy()
            label = (table["kind"].to_numpy()[keep] == "recorded zero, expressing sex").astype(int)
            w = weights[:, keep]
            for name in roots:
                if name == args.reference:
                    continue
                values = {}
                for method in (name, args.reference):
                    s = table[f"score:{method}"].to_numpy()[keep]
                    values[method] = np.array([roc_auc_score(label, s)] + [
                        roc_auc_score(label[row > 0], s[row > 0], sample_weight=row[row > 0])
                        if len(np.unique(label[row > 0])) == 2 else np.nan for row in w])
                difference = values[name] - values[args.reference]
                paired.append({"tissue": tissue, "gene": gene, "method": name, "reference": args.reference,
                               "difference": float(difference[0]), "low": float(np.nanquantile(difference[1:], 0.025)),
                               "high": float(np.nanquantile(difference[1:], 0.975))})
    pd.concat(auroc_frames).to_csv(output / "sex_auroc.csv", index=False, float_format="%.5f")
    pd.concat(fill_frames).to_csv(output / "sex_fill_rates.csv", index=False, float_format="%.5f")
    pd.DataFrame(paired).to_csv(output / "sex_paired.csv", index=False, float_format="%.5f")
    print(pd.concat(auroc_frames).round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(3).to_string(index=False))

def protein(args: argparse.Namespace, roots: dict[str, Path], output: Path) -> None:
    import evaluate_protein_within_state as ps

    benchmark = REPOSITORY / "artifacts/paper_evidence/papalexi_crossmodal/benchmark"
    inputs = ps.load_inputs(REPOSITORY / "external_data/prepared/papalexi_eccite_crossmodal.h5ad", benchmark,
                            REPOSITORY / "artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet")
    adata = ad.read_h5ad(benchmark / "corrupted.h5ad")
    counts = dense(adata.layers["corrupted_counts"])
    split = pd.read_parquet(benchmark / "splits.parquet").set_index("cell_id").loc[adata.obs_names.astype(str), "split"].to_numpy()
    rows, cols = np.where((counts == 0) & (split == "test")[:, None])
    coordinates = pd.read_parquet(benchmark / "coordinates.parquet")
    masked = np.zeros(counts.shape, dtype=bool)
    masked[coordinates["cell_index"].to_numpy(dtype=np.int64), coordinates["gene_index"].to_numpy(dtype=np.int64)] = True
    gene = ps.PROTEIN_TO_GENE["PDL1"]
    column = inputs["gene_index"][gene]
    frame = None
    fill_rows = []
    for name, root in roots.items():
        scores = np.load(root / "papalexi_crossmodal" / "test_scores.npy")
        if len(scores) != len(rows):
            raise ValueError(f"{name}: {len(scores)} scores for {len(rows)} held-out zeros")
        keep = cols == column
        inputs["scores"] = pd.DataFrame({"cell_id": adata.obs_names.astype(str).to_numpy()[rows[keep]], "gene_id": gene,
                                         "selector_score": scores[keep], "masked_positive": masked[rows[keep], cols[keep]].astype(int)})
        current = ps.recorded_zero_frame(inputs, "PDL1", gene)
        if frame is None:
            frame = current.drop(columns=["selector_score"]).rename(columns={"safe_fusion": name})
        else:
            frame[name] = current.set_index("cell_id").loc[frame["cell_id"], "safe_fusion"].to_numpy()
        rank = np.empty(len(scores), dtype=np.int64)
        rank[np.argsort(-scores.astype(np.float64), kind="stable")] = np.arange(len(scores))
        zero_rank = pd.Series(rank, index=pd.MultiIndex.from_arrays([rows, cols])).loc[
            list(zip(frame["cell_index"].to_numpy(dtype=int), np.full(len(frame), column)))].to_numpy()
        for pct in (1, 5, 10):
            fill_rows.append({"method": name, "fill_pct": pct, "n_zeros": int(len(frame)),
                              "filled": int((zero_rank < max(1, int(round(pct / 100 * len(scores))))).sum())})
    rankings = (*roots, "svd", "weighted_knn", "scvi", "library_size", "target_baseline", "ifng_score")
    data = frame.assign(target_baseline=frame["target_baseline_clr"])
    group_names, groups = np.unique(data["target"].astype(str), return_inverse=True)
    reference_first = (args.reference, *[r for r in rankings if r != args.reference])
    job = ps.Job(
        protein="PDL1", gene=gene, scale="clr", cells="recorded_rna_zero", scope="all", resampling="target_clusters",
        rankings=tuple(r if r != args.reference else "safe_fusion" for r in reference_first),
        scores=data[list(reference_first)].to_numpy(dtype=float), response=data["adt_clr"].to_numpy(dtype=float),
        groups=groups, replicate=np.unique(data["replicate"].astype(str), return_inverse=True)[1],
        covariates={name: data[name].to_numpy(dtype=float) for name in ("target_baseline", "ifng_score", "library_size")},
        within=True, covariate_sets={"target_ifng_library": ps.COVARIATE_SETS["target_ifng_library"]},
        seed=(args.seed, list(ps.PROTEIN_TO_GENE).index("PDL1"), 0, 0, 0), draws=args.draws,
        group_names=list(group_names))
    association, paired = ps.run_job(job)
    association["ranking"] = association["ranking"].replace({"safe_fusion": args.reference})
    paired = paired.rename(columns={"safe_fusion_minus_comparator": f"{args.reference}_minus_comparator"})
    association.to_csv(output / "pdl1_association.csv", index=False, float_format="%.5f")
    paired.to_csv(output / "pdl1_paired.csv", index=False, float_format="%.5f")
    pd.DataFrame(fill_rows).to_csv(output / "pdl1_global_fill.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(association.round(3).to_string(index=False))
        print(paired.round(3).to_string(index=False))
        print(pd.DataFrame(fill_rows).to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis", choices=("knockdown", "sex", "protein"), nargs="+", required=True)
    parser.add_argument("--scores", nargs=2, action="append", metavar=("NAME", "ROOT"), required=True,
                        help="Selector name and root with <chain>/test_scores.npy.")
    parser.add_argument("--reference", required=True, help="Selector name that the paired differences subtract.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-detection", type=float, default=0.2)
    parser.add_argument("--depth-strata", type=int, default=5)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    roots = {name: Path(root) for name, root in args.scores}
    if args.reference not in roots:
        parser.error("--reference must be one of the --scores names")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for analysis in args.analysis:
        {"knockdown": knockdown, "sex": sex, "protein": protein}[analysis](args, roots, args.output_dir)
    (args.output_dir / f"design_{'_'.join(args.analysis)}.json").write_text(json.dumps({
        "scores": {name: str(root) for name, root in roots.items()}, "reference": args.reference,
        "draws": args.draws, "seed": args.seed, "depth_strata": args.depth_strata,
        "min_detection": args.min_detection}, indent=2) + "\n")

if __name__ == "__main__":
    main()
