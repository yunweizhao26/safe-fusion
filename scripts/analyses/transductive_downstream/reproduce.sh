#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o artifacts/paper_evidence/review_round4/transductive_downstream/logs/%x-%A_%a.out
#SBATCH -e artifacts/paper_evidence/review_round4/transductive_downstream/logs/%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -u scripts/analyses/transductive_downstream/reproduce.py "$SLURM_ARRAY_TASK_ID"
