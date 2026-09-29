#!/usr/bin/env bash
#SBATCH --job-name=sf-detection-rule
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-7
#SBATCH --output=logs/slurm-detection-rule-%A_%a.out
#SBATCH --error=logs/slurm-detection-rule-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

keys=(colon pancreas_0 pancreas_1 pancreas_2)
key="${keys[SLURM_ARRAY_TASK_ID % ${#keys[@]}]}"
deployment_paths "${key}"
if (( SLURM_ARRAY_TASK_ID < ${#keys[@]} )); then
  data=(--corrupted "${input}" --coordinates "${coordinates}" --splits "${splits}")
  models="${methods_root}"
  output="artifacts/paper_evidence/detection_rule/${key}"
else
  data=(--corrupted "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet")
  models="${deploy_methods}"
  output="${out}/detection_rule"
fi
mapfile -t teacher_args < <(teacher_contract_args "${models}")

.venv/bin/python scripts/calibrated_selective_fill.py \
  "${data[@]}" --truth "${truth}" \
  --fusion-contract "${models}/safe_fusion" "${teacher_args[@]}" \
  --output-dir "${output}" "${fit[@]}" \
  --architecture mlp --budget-mode apply_topk --budgets 0.05 \
  --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
  --detection-rule-mask-rate 0.10 --seed 1729 > /dev/null
