#!/bin/bash
#SBATCH --job-name=lb-masked
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH --output=artifacts/paper_evidence/review_round4/label_baselines/masked/slurm-%A_%a.out
#SBATCH --error=artifacts/paper_evidence/review_round4/label_baselines/masked/slurm-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=8
out=artifacts/paper_evidence/review_round4/label_baselines/masked
sets=(adamson_crispri papalexi_eccite norman_crispra Pancreas Colon)
.venv/bin/python "scripts/analyses/label_baselines/masked/evaluate.py" "${sets[$SLURM_ARRAY_TASK_ID]}"
