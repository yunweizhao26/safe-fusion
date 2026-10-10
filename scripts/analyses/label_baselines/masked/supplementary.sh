#!/bin/bash
#SBATCH --job-name=lb-masked-supp
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=artifacts/paper_evidence/review_round4/label_baselines/masked/supplementary-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
sets=(Pancreas Colon)
.venv/bin/python scripts/analyses/label_baselines/masked/supplementary.py "${sets[$SLURM_ARRAY_TASK_ID]}"
