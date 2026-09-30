#!/usr/bin/env bash
#SBATCH --job-name=sf-r4-sex-colon-prep
#SBATCH --account=torch_pr_634_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-r4-sex-colon-prep-%A_%a.out
#SBATCH --error=logs/slurm-r4-sex-colon-prep-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
OUT2=artifacts/paper_evidence/review_round4/transductive_references/sex_zeros
CF=artifacts/paper_evidence/review_round3/colon_crossfit
fold="${SLURM_ARRAY_TASK_ID}"
.venv/bin/python scripts/build_deployment_inputs.py \
  --truth "${CF}/fold_${fold}/prepared.h5ad" --corrupted "${CF}/fold_${fold}/corrupted.h5ad" \
  --coordinates "${CF}/fold_${fold}/coordinates.parquet" --splits "${CF}/fold_${fold}/splits.parquet" \
  --output-dir "${OUT2}/deployment/colon_${fold}"
