#!/usr/bin/env bash
#SBATCH --job-name=sf-papalexi-scvi
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=logs/slurm-papalexi-scvi-%j.out
#SBATCH --error=logs/slurm-papalexi-scvi-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ROOT=artifacts/external_perturbseq/papalexi_eccite
.conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
  --corrupted "${ROOT}/corrupted.h5ad" \
  --coordinates "${ROOT}/coordinates.parquet" \
  --splits "${ROOT}/splits.parquet" \
  --output artifacts/paper_evidence/papalexi_crossmodal/scvi \
  --epochs 200 \
  --seed 1729
