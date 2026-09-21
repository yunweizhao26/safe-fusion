#!/usr/bin/env bash
#SBATCH --job-name=sf-pap-xm-method
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-pap-xm-method-%j.out
#SBATCH --error=logs/slurm-pap-xm-method-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PY="uv run python"
ROOT=artifacts/paper_evidence/papalexi_crossmodal/benchmark

"${PY}" scripts/run_leakage_safe_method.py \
  --method graph_smooth \
  --input "${ROOT}/corrupted.h5ad" \
  --coordinates "${ROOT}/coordinates.parquet" \
  --splits "${ROOT}/splits.parquet" \
  --output "${ROOT}/graph_smooth" \
  --seed 1729

"${PY}" scripts/run_leakage_safe_method.py \
  --method safe_fusion \
  --input "${ROOT}/corrupted.h5ad" \
  --coordinates "${ROOT}/coordinates.parquet" \
  --splits "${ROOT}/splits.parquet" \
  --output "${ROOT}/safe_fusion" \
  --epochs 12 \
  --batch-size 64 \
  --seed 1729
