#!/usr/bin/env bash
#SBATCH --job-name=sf-baseline-figure
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-baseline-figure-%j.out
#SBATCH --error=logs/slurm-baseline-figure-%j.err

# Masked-F1 curves of the comparators on the count scale, the combined table
# with the Safe Fusion selector and the stacked selectors, paired unit
# intervals, and the paper figure. Run after slurm_rescale_saver_contracts.sh
# and, for the stacked rows, slurm_stacked_selector_baselines.sh.
# FIGURE sets the figure path.
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python scripts/compute_matched_baseline_f1_curves.py
.venv/bin/python scripts/combine_mlp_baseline_curves.py
.venv/bin/python scripts/paired_masked_f1_bootstrap.py
MPLBACKEND=Agg MPLCONFIGDIR="/tmp/sf-f1-figure-${SLURM_JOB_ID:-local}" \
  .venv/bin/python scripts/plot_selector_f1_fillrate.py \
  --input artifacts/paper_evidence/selector_f1_fillrate_mlp_baselines_1000_points.csv \
  --output "${FIGURE:-artifacts/paper_evidence/figures/f1_fillrate_3panel_oup.png}"
