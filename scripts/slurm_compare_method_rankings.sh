#!/usr/bin/env bash
#SBATCH --job-name=sf-rank-compare
#SBATCH --account=torch_pr_634_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-rank-compare-%j.out
#SBATCH --error=logs/slurm-rank-compare-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python scripts/compare_method_rankings.py --output-dir artifacts/paper_evidence/ranking_comparison
