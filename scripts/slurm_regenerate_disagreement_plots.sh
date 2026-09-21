#!/usr/bin/env bash
#SBATCH --job-name=sf-disagreement
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-disagreement-%j.out
#SBATCH --error=logs/slurm-disagreement-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/safefusion-mpl-${SLURM_JOB_ID}

mkdir -p "${MPLCONFIGDIR}" artifacts/paper_evidence/disagreement_observed logs
: "${DISAGREEMENT_INPUT_ROOT:?set DISAGREEMENT_INPUT_ROOT to the analysis input directory}"

uv run python -u scripts/regenerate_legacy_disagreement_plots.py \
  --input-root "$DISAGREEMENT_INPUT_ROOT" \
  --output-dir artifacts/paper_evidence/disagreement_observed \
  --candidate-source observed-h5ad \
  --chunk-rows 64 \
  --only all
