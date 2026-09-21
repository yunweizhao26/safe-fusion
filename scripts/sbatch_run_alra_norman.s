#!/bin/bash
#SBATCH --job-name=sf_alra
#SBATCH --output=logs/sf_alra_%j.log
#SBATCH --error=logs/sf_alra_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

uv run python -u scripts/run_alra_baseline.py \
  --corrupted artifacts/paper_evidence/norman_crispra/corrupted.h5ad \
  --splits artifacts/paper_evidence/norman_crispra/splits.parquet \
  --output artifacts/paper_evidence/baselines/alra/norman_mask_010 \
  --components 100 --power-iterations 2 --quantile-prob 0.001 --seed 1729

echo ALRA_NORMAN_DONE
