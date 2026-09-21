#!/usr/bin/env bash
#SBATCH --job-name=sf-eval-mlp-range
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-evaluate-mlp-biology-range-%A_%a.out
#SBATCH --error=logs/slurm-evaluate-mlp-biology-range-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
CF=artifacts/paper_evidence/pancreas_crossfit
OUT=artifacts/paper_evidence/selector_mlp_biology_range

method_args=()
extra_args=()
for pct in $(seq 1 10); do
  budget="0p0${pct}"
  if (( pct == 10 )); then
    budget="0p1"
  fi
  method_args+=(--method "mlp_${pct}pct=${OUT}/PLACEHOLDER/safe_fusion_calibrated_mlp_topk_${budget}")
  extra_args+=(--extra-method "mlp_${pct}pct=selector_mlp_biology_range/safe_fusion_calibrated_mlp_topk_${budget}")
done

if (( SLURM_ARRAY_TASK_ID == 0 )); then
  COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
  colon_methods=(--method "graph_smooth=${COL}/methods/standardized/colon_epithelial/mask_010/graph_smooth")
  for value in "${method_args[@]}"; do
    colon_methods+=("${value/PLACEHOLDER/colon}")
  done
  "${PY}" scripts/evaluate_colon_donor_biology.py \
    --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" \
    --corrupted "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
    --coordinates "${COL}/data/colon_epithelial/coordinates/mask_010.parquet" \
    --splits "${COL}/data/colon_epithelial/splits.parquet" \
    "${colon_methods[@]}" \
    --output-dir "${OUT}/biology/colon" --bootstrap 2000 --seed 1729
elif (( SLURM_ARRAY_TASK_ID == 1 )); then
  N=artifacts/paper_evidence/norman_crispra
  norman_methods=(--method "graph_smooth=${N}/methods/graph_smooth")
  for value in "${method_args[@]}"; do
    norman_methods+=("${value/PLACEHOLDER/norman_crispra}")
  done
  "${PY}" scripts/evaluate_perturbation_preservation.py \
    --truth external_data/prepared/norman_crispra.h5ad \
    --corrupted "${N}/corrupted.h5ad" --splits "${N}/splits.parquet" \
    "${norman_methods[@]}" \
    --output-dir "${OUT}/biology/norman_crispra" --bootstrap 2000 --seed 1729
else
  "${PY}" scripts/evaluate_pancreas_crossfit_biology.py \
    --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
    --corrupted "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
    --coordinates "${PAN}/data/pancreas_islets/coordinates/mask_010.parquet" \
    --crossfit-dir "${CF}" --safe-fusion-subdir safe_fusion_b32 \
    "${extra_args[@]}" \
    --output-dir "${OUT}/biology/pancreas" --bootstrap 2000 --seed 1729
fi
