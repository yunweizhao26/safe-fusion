#!/usr/bin/env bash
#SBATCH --job-name=sf-stacked-selectors
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-stacked-selectors-%A_%a.out
#SBATCH --error=logs/slurm-stacked-selectors-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

keys=(pancreas_0 pancreas_1 pancreas_2 colon)
key="${keys[SLURM_ARRAY_TASK_ID]}"
.venv/bin/python scripts/stacked_selector_baselines.py --unit "${key}" --seed 1729 ${STACKED_ARGS:-}
