#!/usr/bin/env bash
#SBATCH --job-name=sf-protein-state
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs,cpu_short
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-protein-state-%j.out
#SBATCH --error=logs/slurm-protein-state-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ROOT=artifacts/paper_evidence/papalexi_crossmodal/benchmark
OUT=artifacts/paper_evidence/review_round2/protein
source scripts/unit_paths.sh
mapfile -t teacher_args < <(teacher_contract_args "${ROOT}")
mapfile -t genes < <(.venv/bin/python -c \
  'import sys, anndata; print("\n".join(anndata.read_h5ad(sys.argv[1], backed="r").var_names))' \
  "${ROOT}/corrupted.h5ad")

.venv/bin/python scripts/calibrated_selective_fill.py \
  --corrupted "${ROOT}/corrupted.h5ad" \
  --truth external_data/prepared/papalexi_eccite_crossmodal.h5ad \
  --coordinates "${ROOT}/coordinates.parquet" \
  --splits "${ROOT}/splits.parquet" \
  --fusion-contract "${ROOT}/safe_fusion" \
  "${teacher_args[@]}" \
  --output-dir "${OUT}/global_fill_selector" \
  --fit-split development \
  --architecture mlp \
  --budget-mode apply_topk \
  --budgets 0.01 0.05 0.1 \
  --curve-min-budget 0.001 \
  --curve-max-budget 1.0 \
  --curve-points 1000 \
  --score-gene "${genes[@]}" \
  --seed 1729 > /dev/null

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_protein_within_state.py \
  --benchmark-root "${ROOT}" \
  --global-fill-dir "${OUT}/global_fill_selector" \
  --output-dir "${OUT}/evaluation" \
  --bootstrap 2000 \
  --seed 1729 \
  --workers "${SLURM_CPUS_PER_TASK}"
