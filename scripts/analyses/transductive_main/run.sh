#!/usr/bin/env bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1
exec .venv/bin/python scripts/analyses/transductive_main/"$1" "${@:2}"
