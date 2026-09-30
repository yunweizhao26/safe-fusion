#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH -o artifacts/paper_evidence/review_round4/transductive_downstream/logs/%x-%A_%a.out
#SBATCH -e artifacts/paper_evidence/review_round4/transductive_downstream/logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -u scripts/analyses/transductive_downstream/pipeline.py "$1" "${SLURM_ARRAY_TASK_ID:-0}" "${2:-full}"
