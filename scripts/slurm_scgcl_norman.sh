#!/usr/bin/env bash
#SBATCH --job-name=sf-scgcl-norman
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/slurm-scgcl-norman-%j.out
#SBATCH --error=logs/slurm-scgcl-norman-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv-baselines/bin/python scripts/run_scgcl_baseline.py \
  --corrupted artifacts/paper_evidence/norman_crispra/corrupted.h5ad \
  --output artifacts/paper_evidence/baselines/scgcl/norman \
  --repo baselines_and_data/scGCL \
  --epochs 300 \
  --lr 1e-6 \
  --stability-note "common finite learning rate selected without masked truth" \
  --device cpu \
  --checkpoint-every 25
