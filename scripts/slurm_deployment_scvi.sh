#!/usr/bin/env bash
#SBATCH --job-name=sf-deploy-scvi
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-8
#SBATCH --output=logs/slurm-deployment-scvi-%A_%a.out
#SBATCH --error=logs/slurm-deployment-scvi-%A_%a.err

# Deployment analysis, step 2 of 3. Fits the inductive scVI teacher on the
# hybrid input written by slurm_deployment_prepare.sh. The fitted values differ
# between GPU models, so every scVI fit runs on an L40S.
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

deployment_paths "${DEPLOY_KEYS[SLURM_ARRAY_TASK_ID]}"
.conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
  --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
  --splits "${out}/splits.parquet" --output "${deploy_methods}/scvi_inductive" --seed 1729
echo "step=scvi_inductive elapsed_s=${SECONDS}"
