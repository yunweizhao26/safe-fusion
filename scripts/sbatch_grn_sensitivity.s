#!/bin/bash
#SBATCH --job-name=sf_grn
#SBATCH --output=logs/sf_grn_%j.log
#SBATCH --error=logs/sf_grn_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
: "${SERGIO_ROOT:?set SERGIO_ROOT to a SERGIO checkout}"

uv run python -u scripts/grn_sensitivity_sergio.py \
  --sergio-root "$SERGIO_ROOT" \
  --output-dir artifacts/paper_evidence/grn_sensitivity \
  --n-genes 100 --n-tfs 10 --n-cells 1200 --seed 1729

echo GRN_SENSITIVITY_DONE
