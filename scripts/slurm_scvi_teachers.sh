#!/usr/bin/env bash
#SBATCH --job-name=sf-scvi-teacher
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-9
#SBATCH --output=logs/slurm-scvi-teacher-%A_%a.out
#SBATCH --error=logs/slurm-scvi-teacher-%A_%a.err

# Fit the inductive scVI teacher for one dataset per task (units from
# scripts/unit_paths.sh). The fitted values differ between GPU models, so every
# scVI fit runs on an L40S.
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

unit_paths "${UNITS[SLURM_ARRAY_TASK_ID]}"
.conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
  --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
  --output "${methods_root}/scvi_inductive" --seed 1729
