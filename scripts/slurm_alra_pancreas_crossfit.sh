#!/usr/bin/env bash
#SBATCH --job-name=sf-alra-crossfit
#SBATCH --account=torch_pr_634_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-alra-crossfit-%A_%a.out
#SBATCH --error=logs/slurm-alra-crossfit-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

fold="${SLURM_ARRAY_TASK_ID}"
pancreas=artifacts/pancreas_runs/0b2469810675-45c81b160d78
crossfit=artifacts/paper_evidence/pancreas_crossfit

.venv/bin/python scripts/run_alra_baseline.py \
  --corrupted "${pancreas}/data/pancreas_islets/corrupted/mask_010.h5ad" \
  --splits "${crossfit}/fold_${fold}/splits.parquet" \
  --output "${crossfit}/fold_${fold}/alra" \
  --seed 1729
