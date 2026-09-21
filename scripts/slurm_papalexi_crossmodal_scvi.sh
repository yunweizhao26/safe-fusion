#!/usr/bin/env bash
#SBATCH --job-name=sf-pap-xm-scvi
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/slurm-pap-xm-scvi-%j.out
#SBATCH --error=logs/slurm-pap-xm-scvi-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
ROOT=artifacts/paper_evidence/papalexi_crossmodal/benchmark
.conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
  --corrupted "${ROOT}/corrupted.h5ad" \
  --coordinates "${ROOT}/coordinates.parquet" \
  --splits "${ROOT}/splits.parquet" \
  --output "${ROOT}/scvi" \
  --epochs 200 \
  --seed 1729
