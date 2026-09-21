#!/usr/bin/env bash
#SBATCH --job-name=sf-down-final
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-complete-downstream-final-%j.out
#SBATCH --error=logs/slurm-complete-downstream-final-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/safefusion-downstream-${SLURM_JOB_ID}"

.venv/bin/python scripts/summarize_complete_downstream.py
uv run python scripts/plot_complete_downstream.py
.venv/bin/python scripts/verify_complete_downstream.py
