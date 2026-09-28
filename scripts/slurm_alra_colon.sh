#!/usr/bin/env bash
#SBATCH --job-name=sf-alra-colon
#SBATCH --account=torch_pr_634_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-alra-colon-%j.out
#SBATCH --error=logs/slurm-alra-colon-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
source scripts/unit_paths.sh

unit_paths colon
.venv/bin/python scripts/run_alra_baseline.py \
  --corrupted "${input}" --splits "${splits}" \
  --output "${OUT:-artifacts/paper_evidence/baselines/alra/colon_mask_010}" \
  --components 100 --power-iterations 2 --quantile-prob 0.001 --seed 1729
