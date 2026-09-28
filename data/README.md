# Papalexi RNA-protein evaluation inputs

`papalexi_crossmodal_inputs.tar.gz` holds the inputs of the RNA-protein
analysis of the manuscript (Figure 2 and Supplementary Table S9): the prepared
Papalexi ECCITE-seq RNA and protein matrix, the masked benchmark, the fitted
teachers, fused value, selector and scVI comparator, and the matched RNA and
antibody panel. With these inputs the evaluation runs without regenerating the
upstream chain. The archive is stored with Git LFS.

- Size: 344,659,694 bytes (345 MB)
- SHA-256: `dfdbb72a33e642ee3095ab054dd9bae2209809d76841cb83ccb0d33d62230185`

## Contents

Paths are relative to the repository root.

- `artifacts/paper_evidence/papalexi_crossmodal/audit/`: RNA-protein cell
  matching audit, including `matched_rna_adt_panel.parquet`.
- `artifacts/paper_evidence/papalexi_crossmodal/benchmark/`: `corrupted.h5ad`,
  `splits.parquet` and `coordinates.parquet`; the five teacher contracts
  (`gene_median`, `svd_impute`, `graph_smooth`, `magic_inductive`,
  `scvi_inductive`); the fused value (`safe_fusion`); the transductive scVI
  comparator (`scvi`); the selector output
  (`mlp_selector/selected_gene_scores.parquet`); and the evaluation outputs of
  the manuscript (`evaluation_cd274`, `evaluation_cd274_raw_counts`,
  `evaluation_pdl1_state`, `evaluation`, `evaluation_raw_counts`) for
  comparison.
- `artifacts/paper_evidence/papalexi_crossmodal/benchmark/evaluation_quick`
  and the directories `mlp_selector/`, `scvi/` and `samuel_reproduction/`
  directly under `artifacts/paper_evidence/papalexi_crossmodal/`: outputs of
  earlier runs, kept for reference and not read by the launchers.
- `external_data/prepared/papalexi_eccite_crossmodal.h5ad` and its
  `.report.json`: the prepared RNA and protein matrix read by the evaluation.

## Use

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
`artifacts/paper_evidence/papalexi_crossmodal/benchmark` and writes Figure 2
to `artifacts/paper_evidence/figures`. `PAPER_DIR` sets the directory of an
extra PNG copy of the figure. `scripts/slurm_evaluate_papalexi_crossmodal.sh`
and `scripts/slurm_evaluate_papalexi_crossmodal_raw.sh` run from the same
inputs. They evaluate CD86, PD-L2 and TIM-3 together with CD274, on centered
log ratio (the last three columns of Table S9) and raw antibody counts.
