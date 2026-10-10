#!/bin/bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
O=artifacts/paper_evidence/review_round4/current_tables
export TMPDIR=$PWD/$O/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
if [[ "$1" == extra ]]; then shift; .venv/bin/python "scripts/analyses/current_tables/extra.py" "$@"; else .venv/bin/python "scripts/analyses/current_tables/tables.py" "$@"; fi
