#!/usr/bin/env bash
#SBATCH --job-name=sf-stack
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-9
#SBATCH --output=logs/slurm-stack-%A_%a.out
#SBATCH --error=logs/slurm-stack-%A_%a.err

# Fit the Safe Fusion fused value from all five teachers for one dataset per
# task (units from scripts/unit_paths.sh). VALUE_MODEL selects the value model
# of run_leakage_safe_method.py: boosted (default) writes the production value
# to <methods_root>/safe_fusion, and linear writes the linear combination of
# the teachers (one least-squares weight per teacher) to
# <methods_root>/safe_fusion_linear.
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
VALUE_MODEL="${VALUE_MODEL:-boosted}"

unit_paths "${UNITS[SLURM_ARRAY_TASK_ID]}"
mapfile -t teacher_args < <(teacher_contract_args "${methods_root}")
.venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion \
  --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
  --output "${methods_root}/$(fused_value_contract "${VALUE_MODEL}")" \
  --value-model "${VALUE_MODEL}" --seed 1729 "${teacher_args[@]}"
