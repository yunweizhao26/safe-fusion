#!/usr/bin/env bash
#SBATCH --job-name=sf-kd-eval
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-kd-eval-%j.out
#SBATCH --error=logs/slurm-kd-eval-%j.err

set -euo pipefail
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh
source scripts/knockdown_paths.sh
knockdown_paths norman_crispra
LEAKAGE_FREE="$(dirname "${NORMAN_REBUILT}")"

.venv/bin/python scripts/evaluate_condition_masked_f1.py --seed 1729 \
  --norman-benchmark "${NORMAN_REBUILT}" --norman-methods "${methods_root}" \
  --norman-label-methods "${label_root}" --norman-truth "${truth}" \
  --norman-selector "${LEAKAGE_FREE}/selector_mlp_biology_range_fullteachers/norman_crispra" \
  --output-dir "${KNOCKDOWN_ROOT}/masked_f1" --unit-output-dir "${KNOCKDOWN_ROOT}/masked_f1"
.venv/bin/python scripts/evaluate_knockdown_zero_analyses.py --depth-strata 5 --seed 1729 \
  --review-root "${KNOCKDOWN_ROOT}" --norman-benchmark "${NORMAN_REBUILT}" --norman-root "${NORMAN_KNOCKDOWN}" \
  --output-dir "${KNOCKDOWN_ROOT}/evaluation"
