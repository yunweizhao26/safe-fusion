#!/usr/bin/env bash
#SBATCH --job-name=sf-benchmark-data
#SBATCH --account=torch_pr_634_general
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-benchmark-data-%j.out
#SBATCH --error=logs/slurm-benchmark-data-%j.err

# Count cells, genes, test candidates and masked positives of every masked
# benchmark in scripts/unit_paths.sh (Supplementary Table S2).
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
source scripts/unit_paths.sh

unit_args=()
for key in "${UNITS[@]}"; do
  unit_paths "${key}"
  unit_args+=(--unit "${key}" "${input}" "${coordinates}" "${splits}")
done
.venv/bin/python scripts/summarize_benchmark_data.py "${unit_args[@]}" \
  --output "${OUTPUT:-artifacts/paper_evidence/benchmark_data/benchmark_data.csv}"
