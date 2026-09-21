#!/usr/bin/env bash
#SBATCH --job-name=sf-bio-figures
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-bio-figures-%j.out
#SBATCH --error=logs/slurm-bio-figures-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/safefusion-bio-figures-${SLURM_JOB_ID}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
uv run python scripts/plot_biological_range_figures.py --pdl1-only
