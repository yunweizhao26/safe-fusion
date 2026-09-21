#!/usr/bin/env bash
#SBATCH --job-name=sf-pap-xm-mlp
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-pap-xm-mlp-%j.out
#SBATCH --error=logs/slurm-pap-xm-mlp-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ROOT=artifacts/paper_evidence/papalexi_crossmodal/benchmark
.venv/bin/python scripts/calibrated_selective_fill.py \
  --corrupted "${ROOT}/corrupted.h5ad" \
  --truth external_data/prepared/papalexi_eccite_crossmodal.h5ad \
  --coordinates "${ROOT}/coordinates.parquet" \
  --splits "${ROOT}/splits.parquet" \
  --fusion-contract "${ROOT}/safe_fusion" \
  --teacher-contract "${ROOT}/graph_smooth" \
  --output-dir "${ROOT}/mlp_selector" \
  --fit-split development \
  --architecture mlp \
  --budget-mode apply_topk \
  --budgets 0.02 \
  --curve-min-budget 0.001 \
  --curve-max-budget 1.0 \
  --curve-points 1000 \
  --score-gene CD274 CD86 PDCD1LG2 HAVCR2 \
  --seed 1729
