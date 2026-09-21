#!/usr/bin/env bash
#SBATCH --job-name=sf-pair-mlp-attr
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-pair-mlp-attribution-range-%A_%a.out
#SBATCH --error=logs/slurm-pair-mlp-attribution-range-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

PY=.venv/bin/python
ROOT=artifacts/paper_evidence/selector_mlp_attribution_range

if (( SLURM_ARRAY_TASK_ID == 0 )); then
  "${PY}" scripts/combine_selector_attribution.py \
    --input-dir "${ROOT}/pancreas/fold_0" \
    --input-dir "${ROOT}/pancreas/fold_1" \
    --input-dir "${ROOT}/pancreas/fold_2" \
    --output-dir "${ROOT}/paired/pancreas" \
    --reference-selector full__mlp --bootstrap 2000 --seed 1729
elif (( SLURM_ARRAY_TASK_ID == 1 )); then
  "${PY}" scripts/combine_selector_attribution.py \
    --input-dir "${ROOT}/colon" \
    --output-dir "${ROOT}/paired/colon" \
    --reference-selector full__mlp --bootstrap 2000 --seed 1729
else
  "${PY}" scripts/combine_selector_attribution.py \
    --input-dir "${ROOT}/norman_crispra" \
    --output-dir "${ROOT}/paired/norman_crispra" \
    --reference-selector full__mlp --bootstrap 2000 --seed 1729
fi
