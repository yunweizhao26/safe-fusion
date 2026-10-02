# Papalexi RNA-protein evaluation inputs

Both archives are stored with Git LFS. Paths inside them are relative to the
repository root.

## Current protein analysis: Figure 2 and Supplementary Tables S19 and S20

`papalexi_protein_current_inputs.tar.gz` contains the saved inputs for
`scripts/analyses/transductive_references/protein_cs.sh` in its `evaluate` and
`plot` stages. No model fitting is needed. See the quick path in
[section 22 of the reproduction guide](../docs/REPRODUCTION.md#22-perturbation-sex-and-protein-controls).

- Size: 443,164,880 bytes (443.2 MB)
- SHA-256: `ff1082d83c9f67a578fa0304d5cc6838ce9df099cdedc66c38afb72db53be932`
- Contents: 43 regular files. Symlinks are resolved. Fitted model
  weights, logs, precomputed evaluation tables and figures are excluded.

Under `artifacts/paper_evidence/review_round4/transductive_references/protein_cs/`:

- `corrupted.h5ad`, `splits.parquet` and `coordinates.parquet`: masked RNA,
  cell splits and mask coordinates.
- `{gene_median,svd_impute,graph_smooth,magic,scvi_counts}/`: the five
  transductive teacher contracts, each with `mean.npy` and `metadata.json`.
- `safe_fusion/`: fused values (`mean.npy`) and contract metadata.
- `magic_standard/` and `scvi/`: standard outputs (`mean.npy` and
  `metadata.json`). The state-baseline evaluator reads this local scVI copy.
- `mlp_selector/`: `selected_gene_scores.parquet`,
  `selected_gene_scores_metadata.json`, `calibration_report.json`, and the
  `mean.npy` and `metadata.json` of
  `safe_fusion_calibrated_mlp_topk_0p02/`.
- `global_fill_selector/`: the same three score and calibration files,
  plus `mean.npy` and `metadata.json` in
  `safe_fusion_calibrated_mlp_topk_{0p01,0p05,0p1}/` for the 1%, 5% and 10%
  global fills.

These protein selectors use saved rankings and fixed fill fractions. They
do not enable the isotonic detection rule, so this input set has no separate
isotonic outputs.

The remaining inputs retain their original paths:

- `external_data/prepared/papalexi_eccite_crossmodal.h5ad` and
  `papalexi_eccite_crossmodal.report.json`: prepared RNA and protein data and
  preparation report. Cell annotations and counts supply the state baselines.
- `artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet`:
  matched cells, proteins, replicates and perturbation targets.
- `artifacts/paper_evidence/papalexi_crossmodal/benchmark/{svd_impute,graph_smooth,scvi}/`:
  standard SVD, weighted kNN and scVI outputs (`mean.npy` and `metadata.json`)
  read by the crossmodal and matched within-state evaluators.
- `artifacts/paper_evidence/review_round2/protein/global_fill_selector/selected_gene_scores.parquet`:
  the saved cell-zero-score control read by `protein_matched.py`.

Download and unpack from the repository root on a compute node:

```bash
git lfs install
git lfs pull --include "data/papalexi_protein_current_inputs.tar.gz"
echo "ff1082d83c9f67a578fa0304d5cc6838ce9df099cdedc66c38afb72db53be932  data/papalexi_protein_current_inputs.tar.gz" | sha256sum --check
tar -xzf data/papalexi_protein_current_inputs.tar.gz
```

Run the evaluation-only commands in section 22 to write Figure 2 and the
CSV files for Supplementary Tables S19 and S20. Expected Safe Fusion pooled
Spearman is 0.547 and mean AUROC is 76.5%. The saved paired difference from
SVD is 0.039 [0.006, 0.080].

## Earlier inductive protein analysis

`papalexi_crossmodal_inputs.tar.gz` holds the inputs of the earlier inductive
protein analysis (Safe Fusion pooled Spearman 0.530), not the manuscript's
Figure 2 or Supplementary Tables S19 and S20. It contains the prepared Papalexi ECCITE-seq RNA and protein matrix, the masked benchmark, the
fitted teachers, fused value, selector and scVI comparator, and the matched RNA
and antibody panel. With these inputs the evaluation runs without regenerating the
upstream chain. The archive is stored with Git LFS.

- Size: 344,659,694 bytes (345 MB)
- SHA-256: `dfdbb72a33e642ee3095ab054dd9bae2209809d76841cb83ccb0d33d62230185`

### Contents

Paths are relative to the repository root.

- `artifacts/paper_evidence/papalexi_crossmodal/audit/`: RNA-protein cell
  matching audit, including `matched_rna_adt_panel.parquet`.
- `artifacts/paper_evidence/papalexi_crossmodal/benchmark/`: `corrupted.h5ad`,
  `splits.parquet` and `coordinates.parquet`; the five teacher contracts
  (`gene_median`, `svd_impute`, `graph_smooth`, `magic_inductive`,
  `scvi_inductive`); the fused value (`safe_fusion`); the transductive scVI
  comparator (`scvi`); the selector output
  (`mlp_selector/selected_gene_scores.parquet`); and the earlier inductive
  evaluation outputs (`evaluation_cd274`, `evaluation_cd274_raw_counts`,
  `evaluation_pdl1_state`, `evaluation`, `evaluation_raw_counts`) for
  comparison.
- `artifacts/paper_evidence/papalexi_crossmodal/benchmark/evaluation_quick`
  and the directories `mlp_selector/`, `scvi/` and `samuel_reproduction/`
  directly under `artifacts/paper_evidence/papalexi_crossmodal/`: outputs of
  earlier runs, kept for reference and not read by the launchers.
- `external_data/prepared/papalexi_eccite_crossmodal.h5ad` and its
  `.report.json`: the prepared RNA and protein matrix read by the evaluation.

### Use

From the repository root:

```bash
git lfs install
git lfs pull --include "data/papalexi_crossmodal_inputs.tar.gz"
echo "dfdbb72a33e642ee3095ab054dd9bae2209809d76841cb83ccb0d33d62230185  data/papalexi_crossmodal_inputs.tar.gz" | sha256sum --check
tar -xzf data/papalexi_crossmodal_inputs.tar.gz
mkdir -p logs
sbatch --account=<account> scripts/slurm_evaluate_papalexi_cd274.sh
```

The launcher needs the `.venv` environment and, for its final plotting step,
`.venv-baselines` (see [docs/REPRODUCTION.md](../docs/REPRODUCTION.md)). It
rewrites `evaluation_cd274`, `evaluation_cd274_raw_counts` and
`evaluation_pdl1_state` under
`artifacts/paper_evidence/papalexi_crossmodal/benchmark` and writes the earlier
inductive protein plot to `artifacts/paper_evidence/figures`. `PAPER_DIR` sets the directory of an
extra PNG copy of the figure. `scripts/slurm_evaluate_papalexi_crossmodal.sh`
and `scripts/slurm_evaluate_papalexi_crossmodal_raw.sh` run from the same
inputs. They evaluate CD86, PD-L2 and TIM-3 together with CD274, on centered
log ratio (the additional protein comparisons) and raw antibody counts.
`scripts/slurm_evaluate_protein_within_state.sh` also runs from these inputs.
It refits the selector and writes the earlier within-state evaluation to
`artifacts/paper_evidence/review_round2/protein/evaluation`.
