#!/usr/bin/env bash
#SBATCH --job-name=sf-down-selectors
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7%4
#SBATCH --output=logs/slurm-complete-downstream-selectors-%A_%a.out
#SBATCH --error=logs/slurm-complete-downstream-selectors-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
keys=(pancreas_0 pancreas_1 pancreas_2 adamson_crispri dixit_ko papalexi_eccite zebrafish colon)
key="${keys[SLURM_ARRAY_TASK_ID]}"
unit_paths "${key}"
fit=(--fit-split development)
case "${key}" in
  colon)
    output=artifacts/paper_evidence/selector_mlp_biology_range/colon
    fit=(--fit-split validation --fit-cells 3852)
    ;;
  pancreas_*) output="${methods_root}/selector_mlp_biology_range_fullteachers" ;;
  *) output="${methods_root}/selector_mlp_biology_range" ;;
esac
mapfile -t teacher_args < <(teacher_contract_args "${methods_root}")

.venv/bin/python scripts/calibrated_selective_fill.py \
  --corrupted "${input}" \
  --truth "${truth}" \
  --coordinates "${coordinates}" \
  --splits "${splits}" \
  --fusion-contract "${methods_root}/safe_fusion" \
  "${teacher_args[@]}" \
  --output-dir "${output}" \
  "${fit[@]}" \
  --architecture mlp \
  --budget-mode apply_topk \
  --budgets "${BUDGETS[@]}" \
  --curve-min-budget 0.001 \
  --curve-max-budget 1.0 \
  --curve-points 1000 \
  --seed 1729
