#!/usr/bin/env bash
#SBATCH --job-name=sf-mlp-attr-range
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-4
#SBATCH --output=logs/slurm-selector-mlp-attribution-range-%A_%a.out
#SBATCH --error=logs/slurm-selector-mlp-attribution-range-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
CF=artifacts/paper_evidence/pancreas_crossfit
OUT=artifacts/paper_evidence/selector_mlp_attribution_range
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

common=(
  --variants full no_fusion
  --architectures mlp
  --budgets "${BUDGETS[@]}"
  --max-fit-rows 2000000
  --bootstrap 2000
  --seed 1729
)

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  fold="${SLURM_ARRAY_TASK_ID}"
  "${PY}" scripts/selector_attribution.py \
    --corrupted "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
    --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
    --coordinates "${PAN}/data/pancreas_islets/coordinates/mask_010.parquet" \
    --splits "${CF}/fold_${fold}/splits.parquet" \
    --fusion-contract "${CF}/fold_${fold}/safe_fusion_b32" \
    --teacher-contract "${CF}/fold_${fold}/graph_smooth" \
    --output-dir "${OUT}/pancreas/fold_${fold}" \
    --fit-split development --unit-column donor \
    "${common[@]}"
elif (( SLURM_ARRAY_TASK_ID == 3 )); then
  N=artifacts/paper_evidence/norman_crispra
  "${PY}" scripts/selector_attribution.py \
    --corrupted "${N}/corrupted.h5ad" \
    --truth external_data/prepared/norman_crispra.h5ad \
    --coordinates "${N}/coordinates.parquet" \
    --splits "${N}/splits.parquet" \
    --fusion-contract "${N}/methods/safe_fusion" \
    --teacher-contract "${N}/methods/graph_smooth" \
    --output-dir "${OUT}/norman_crispra" \
    --fit-split development --unit-column target \
    "${common[@]}"
else
  COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
  "${PY}" scripts/selector_attribution.py \
    --corrupted "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
    --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" \
    --coordinates "${COL}/data/colon_epithelial/coordinates/mask_010.parquet" \
    --splits "${COL}/data/colon_epithelial/splits.parquet" \
    --fusion-contract "${COL}/methods/standardized/colon_epithelial/mask_010/safe_fusion" \
    --teacher-contract "${COL}/methods/standardized/colon_epithelial/mask_010/graph_smooth" \
    --teacher-contract "${COL}/methods/standardized/colon_epithelial/mask_010/svd_impute" \
    --teacher-contract "${COL}/methods/standardized/colon_epithelial/mask_010/gene_median" \
    --output-dir "${OUT}/colon" \
    --fit-split validation --unit-column donor \
    "${common[@]}"
fi
