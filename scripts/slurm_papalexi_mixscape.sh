#!/usr/bin/env bash
#SBATCH --job-name=sf-mixscape
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-mixscape-%j.out
#SBATCH --error=logs/slurm-mixscape-%j.err

set -euo pipefail
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
.venv-pertpy/bin/python scripts/run_papalexi_mixscape.py --seed 0
