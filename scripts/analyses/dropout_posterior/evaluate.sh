#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --output=artifacts/paper_evidence/review_round4/dropout_posterior/logs/eval-%A_%a.out
#SBATCH --error=artifacts/paper_evidence/review_round4/dropout_posterior/logs/eval-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} PYTHONDONTWRITEBYTECODE=1
export NUMBA_CACHE_DIR=artifacts/paper_evidence/review_round4/dropout_posterior/runtime/numba-$SLURM_JOB_ID
export MPLCONFIGDIR=artifacts/paper_evidence/review_round4/dropout_posterior/runtime/mpl-$SLURM_JOB_ID
case "$1" in
  masked|thinning|known|sex|protein) ;;
  *) echo "Unknown manuscript evaluator: $1" >&2; exit 2 ;;
esac
.venv/bin/python -u "scripts/analyses/dropout_posterior/$1.py" "${@:2}"
