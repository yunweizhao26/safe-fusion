#!/usr/bin/env bash
#SBATCH --job-name=sf-pert-zeros
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-perturbation-zeros-%j.out
#SBATCH --error=logs/slurm-perturbation-zeros-%j.err

set -euo pipefail
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
.venv/bin/python scripts/evaluate_perturbation_zeros.py --seed 1729
.venv/bin/python scripts/evaluate_condition_masked_f1.py --seed 1729
