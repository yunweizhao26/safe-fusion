#!/usr/bin/env bash
#SBATCH --job-name=sf-baseline-curves
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-baseline-curves-%j.out
#SBATCH --error=logs/slurm-baseline-curves-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python scripts/compute_matched_baseline_f1_curves.py
.venv/bin/python scripts/combine_mlp_baseline_curves.py
.venv/bin/python scripts/paired_masked_f1_bootstrap.py
