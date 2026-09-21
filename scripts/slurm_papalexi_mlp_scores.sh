#!/usr/bin/env bash
#SBATCH --job-name=sf-papalexi-mlp
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-papalexi-mlp-%j.out
#SBATCH --error=logs/slurm-papalexi-mlp-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ROOT=artifacts/external_perturbseq/papalexi_eccite
.venv/bin/python scripts/calibrated_selective_fill.py \
  --corrupted "${ROOT}/corrupted.h5ad" \
  --truth external_data/prepared/papalexi_eccite.h5ad \
  --coordinates "${ROOT}/coordinates.parquet" \
  --splits "${ROOT}/splits.parquet" \
  --fusion-contract "${ROOT}/safe_fusion" \
  --teacher-contract "${ROOT}/graph_smooth" \
  --output-dir artifacts/paper_evidence/papalexi_crossmodal/mlp_selector \
  --fit-split development \
  --architecture mlp \
  --budget-mode apply_topk \
  --budgets 0.02 \
  --curve-min-budget 0.001 \
  --curve-max-budget 1.0 \
  --curve-points 1000 \
  --score-gene CD274 CD86 PDCD1LG2 HAVCR2 \
  --seed 1729
