#!/usr/bin/env bash
#SBATCH --job-name=sf-mlp-attr-range
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-selector-mlp-attribution-range-%A_%a.out
#SBATCH --error=logs/slurm-selector-mlp-attribution-range-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
OUT="${OUT:-artifacts/paper_evidence/selector_mlp_attribution_range}"
read -r -a VARIANTS <<< "${VARIANTS:-full teacher_only context_only}"
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

common=(
  --variants "${VARIANTS[@]}"
  --architectures mlp
  --budgets "${BUDGETS[@]}"
  --max-fit-rows 2000000
  --bootstrap 2000
  --seed 1729
)

source scripts/unit_paths.sh
keys=(pancreas_0 pancreas_1 pancreas_2 colon)
key="${keys[SLURM_ARRAY_TASK_ID]}"
unit_paths "${key}"
case "${key}" in
  pancreas_*) output="${OUT}/pancreas/fold_${key##*_}"; fit=(--fit-split development --unit-column donor) ;;
  colon) output="${OUT}/colon"; fit=(--fit-split validation --unit-column donor) ;;
esac
mapfile -t teacher_args < <(teacher_contract_args "${methods_root}")
"${PY}" scripts/selector_attribution.py \
  --corrupted "${input}" \
  --truth "${truth}" \
  --coordinates "${coordinates}" \
  --splits "${splits}" \
  --fusion-contract "${methods_root}/safe_fusion" \
  "${teacher_args[@]}" \
  --output-dir "${output}" \
  "${fit[@]}" \
  "${common[@]}"
