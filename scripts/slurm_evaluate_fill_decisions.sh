#!/usr/bin/env bash
#SBATCH --job-name=sf-fill-decisions
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-fill-decisions-%j.out
#SBATCH --error=logs/slurm-fill-decisions-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SOURCE="${SOURCE:-external_data/cellxgene/63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad}"
IMPUTED_ROOT="${IMPUTED_ROOT:-artifacts/paper_evidence/standard_imputers}"
OUT="${OUT:-artifacts/paper_evidence/fill_decisions}"

.venv/bin/python scripts/evaluate_fill_decisions.py \
  --input "${SOURCE}" --imputed-root "${IMPUTED_ROOT}" --output-dir "${OUT}"
