#!/usr/bin/env bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=artifacts/paper_evidence/review_round4/sex_zero_comparators/logs/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
tissues=(pancreas colon)
.venv/bin/python scripts/analyses/sex_zero_comparators/evaluate.py --tissue "${tissues[SLURM_ARRAY_TASK_ID]}" --stage "${STAGE}"
