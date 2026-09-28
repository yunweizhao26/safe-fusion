#!/usr/bin/env bash
#SBATCH --job-name=sf-saver-norman
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/slurm-saver-norman-%j.out
#SBATCH --error=logs/slurm-saver-norman-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python scripts/run_saver_baseline.py \
  --corrupted artifacts/paper_evidence/norman_crispra/corrupted.h5ad \
  --splits artifacts/paper_evidence/norman_crispra/splits.parquet \
  --output artifacts/paper_evidence/baselines/saver/norman_full_mask_010 \
  --ncores "${SLURM_CPUS_PER_TASK}"
