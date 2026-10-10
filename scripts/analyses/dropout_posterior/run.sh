#!/bin/bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=artifacts/paper_evidence/review_round4/dropout_posterior/logs/fit-%A_%a.out
#SBATCH --error=artifacts/paper_evidence/review_round4/dropout_posterior/logs/fit-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONDONTWRITEBYTECODE=1
.venv/bin/python scripts/analyses/dropout_posterior/posterior.py "${SLURM_ARRAY_TASK_ID:-check}"
