#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/artifacts/paper_evidence/review_round4/reviewer_extras/.mpl"
export NUMBA_CACHE_DIR="$PWD/artifacts/paper_evidence/review_round4/reviewer_extras/.numba"
exec "$@"
