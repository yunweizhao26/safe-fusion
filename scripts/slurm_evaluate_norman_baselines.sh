#!/usr/bin/env bash
#SBATCH --job-name=sf-eval-norman
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-1
#SBATCH --output=logs/slurm-eval-norman-%A_%a.out
#SBATCH --error=logs/slurm-eval-norman-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
norman=artifacts/paper_evidence/norman_crispra

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  method=artifacts/paper_evidence/baselines/saver/norman_full_mask_010
else
  method=artifacts/paper_evidence/baselines/scgcl/norman
fi

uv run python scripts/evaluate_baseline_contract.py \
  --corrupted "${norman}/corrupted.h5ad" \
  --coordinates "${norman}/coordinates.parquet" \
  --splits "${norman}/splits.parquet" \
  --method-output "${method}" \
  --output "${method}/evaluation_full.json"
