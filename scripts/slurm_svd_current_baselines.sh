#!/usr/bin/env bash
#SBATCH --job-name=sf-svd-current
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-svd-current-%A_%a.out
#SBATCH --error=logs/slurm-svd-current-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

pancreas=artifacts/pancreas_runs/0b2469810675-45c81b160d78
crossfit=artifacts/paper_evidence/pancreas_crossfit
norman=artifacts/paper_evidence/norman_crispra

if [[ "${SLURM_ARRAY_TASK_ID}" -le 2 ]]; then
  fold="${SLURM_ARRAY_TASK_ID}"
  input="${pancreas}/data/pancreas_islets/corrupted/mask_010.h5ad"
  coordinates="${pancreas}/data/pancreas_islets/coordinates/mask_010.parquet"
  splits="${crossfit}/fold_${fold}/splits.parquet"
  output="${crossfit}/fold_${fold}/svd_impute"
else
  input="${norman}/corrupted.h5ad"
  coordinates="${norman}/coordinates.parquet"
  splits="${norman}/splits.parquet"
  output="${norman}/methods/svd_impute"
fi

uv run python scripts/run_leakage_safe_method.py \
  --method svd_impute \
  --input "${input}" \
  --coordinates "${coordinates}" \
  --splits "${splits}" \
  --output "${output}" \
  --components 50 \
  --seed 1729
