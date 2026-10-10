#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=artifacts/paper_evidence/review_round4/label_baselines/sex/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TISSUES=(pancreas colon)
.venv/bin/python scripts/analyses/label_baselines/sex/evaluate.py --tissue "${TISSUES[SLURM_ARRAY_TASK_ID]}" --stage "${STAGE}"
