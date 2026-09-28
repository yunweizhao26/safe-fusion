#!/usr/bin/env bash
#SBATCH --job-name=sf-pap-cd274
#SBATCH --account=torch_pr_634_general
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-pap-cd274-%j.out
#SBATCH --error=logs/slurm-pap-cd274-%j.err

# CD274 RNA zeros against surface PD-L1, on CLR and raw antibody counts, the
# perturbation-state baselines and partial correlations, then the paper figure.
# Run after scripts/slurm_papalexi_crossmodal_mlp.sh. PAPER_DIR sets the
# directory of the extra PNG copy of the figure (default: the figure directory).
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ROOT=artifacts/paper_evidence/papalexi_crossmodal/benchmark

for column in adt_clr adt_count; do
  output="${ROOT}/evaluation_cd274"
  [[ "${column}" == adt_count ]] && output="${ROOT}/evaluation_cd274_raw_counts"
  .venv/bin/python scripts/evaluate_papalexi_crossmodal.py \
    --prepared external_data/prepared/papalexi_eccite_crossmodal.h5ad \
    --corrupted "${ROOT}/corrupted.h5ad" \
    --panel artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet \
    --selector-scores "${ROOT}/mlp_selector/selected_gene_scores.parquet" \
    --fusion-mean "${ROOT}/safe_fusion/mean.npy" \
    --graph-mean "${ROOT}/graph_smooth/mean.npy" \
    --svd-mean "${ROOT}/svd_impute/mean.npy" \
    --scvi-mean "${ROOT}/scvi/mean.npy" \
    --output-dir "${output}" \
    --protein-column "${column}" \
    --gene CD274 \
    --bootstrap 1000 \
    --permutations 1000 \
    --seed 1729 > /dev/null
done

.venv/bin/python scripts/evaluate_pdl1_state_baselines.py \
  --benchmark-root "${ROOT}" \
  --output-dir "${ROOT}/evaluation_pdl1_state" \
  --bootstrap 1000 \
  --seed 1729 > /dev/null

MPLBACKEND=Agg MPLCONFIGDIR="/tmp/sf-pdl1-${SLURM_JOB_ID}" \
  .venv-baselines/bin/python scripts/plot_biological_range_figures.py \
  --pdl1-only \
  --paper-dir "${PAPER_DIR:-artifacts/paper_evidence/figures}"
