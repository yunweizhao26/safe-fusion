#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/artifacts/paper_evidence/review_round4/fill_grid/.mpl"
export NUMBA_CACHE_DIR="$PWD/artifacts/paper_evidence/review_round4/fill_grid/.numba"
export XDG_CACHE_HOME="$PWD/artifacts/paper_evidence/review_round4/fill_grid/.cache"
exec .venv/bin/python -u scripts/analyses/fill_grid/run.py "$@"
