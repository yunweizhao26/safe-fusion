#!/usr/bin/env bash
#SBATCH --job-name=sf-teachers
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-35
#SBATCH --output=logs/slurm-teachers-%A_%a.out
#SBATCH --error=logs/slurm-teachers-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

methods=(gene_median svd_impute graph_smooth magic_inductive)
unit_paths "${UNITS[SLURM_ARRAY_TASK_ID / 4]}"
method="${methods[SLURM_ARRAY_TASK_ID % 4]}"
output="${methods_root}/${method}"

if [[ "${method}" == magic_inductive ]]; then
  .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
    --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
    --output "${output}" --seed 1729 --n-jobs "${SLURM_CPUS_PER_TASK}"
else
  .venv/bin/python scripts/run_leakage_safe_method.py --method "${method}" \
    --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
    --output "${output}" --seed 1729
fi
