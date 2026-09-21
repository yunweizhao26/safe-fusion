#!/usr/bin/env bash
#SBATCH --job-name=sf-selector-bio
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-selector-bio-%A_%a.out
#SBATCH --error=logs/slurm-selector-bio-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
CF=artifacts/paper_evidence/pancreas_crossfit
OUT=artifacts/paper_evidence/selector_deployment

if (( SLURM_ARRAY_TASK_ID == 0 )); then
  COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
  ROOT="${OUT}/colon"
  "${PY}" scripts/evaluate_colon_donor_biology.py \
    --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" \
    --corrupted "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
    --coordinates "${COL}/data/colon_epithelial/coordinates/mask_010.parquet" \
    --splits "${COL}/data/colon_epithelial/splits.parquet" \
    --method graph_smooth="${COL}/methods/standardized/colon_epithelial/mask_010/graph_smooth" \
    --method logistic_2pct="${ROOT}/logistic/safe_fusion_calibrated_0p02" \
    --method logistic_5pct="${ROOT}/logistic/safe_fusion_calibrated_0p05" \
    --method logistic_10pct="${ROOT}/logistic/safe_fusion_calibrated_0p1" \
    --method hist_gbdt_2pct="${ROOT}/hist_gbdt/safe_fusion_calibrated_hist_gbdt_0p02" \
    --method mlp_2pct="${ROOT}/mlp/safe_fusion_calibrated_mlp_0p02" \
    --method mlp_5pct="${ROOT}/mlp/safe_fusion_calibrated_mlp_0p05" \
    --method mlp_10pct="${ROOT}/mlp/safe_fusion_calibrated_mlp_0p1" \
    --method rank_ensemble_2pct="${ROOT}/rank_ensemble/safe_fusion_calibrated_rank_ensemble_0p02" \
    --output-dir "${OUT}/biology/colon" --bootstrap 2000 --seed 1729
elif (( SLURM_ARRAY_TASK_ID == 1 )); then
  ROOT="${OUT}/norman_crispra"
  N=artifacts/paper_evidence/norman_crispra
  "${PY}" scripts/evaluate_perturbation_preservation.py \
    --truth external_data/prepared/norman_crispra.h5ad \
    --corrupted "${N}/corrupted.h5ad" --splits "${N}/splits.parquet" \
    --method graph_smooth="${N}/methods/graph_smooth" \
    --method logistic_2pct="${ROOT}/logistic/safe_fusion_calibrated_0p02" \
    --method logistic_5pct="${ROOT}/logistic/safe_fusion_calibrated_0p05" \
    --method logistic_10pct="${ROOT}/logistic/safe_fusion_calibrated_0p1" \
    --method hist_gbdt_2pct="${ROOT}/hist_gbdt/safe_fusion_calibrated_hist_gbdt_0p02" \
    --method mlp_2pct="${ROOT}/mlp/safe_fusion_calibrated_mlp_0p02" \
    --method mlp_5pct="${ROOT}/mlp/safe_fusion_calibrated_mlp_0p05" \
    --method mlp_10pct="${ROOT}/mlp/safe_fusion_calibrated_mlp_0p1" \
    --method rank_ensemble_2pct="${ROOT}/rank_ensemble/safe_fusion_calibrated_rank_ensemble_0p02" \
    --output-dir "${OUT}/biology/norman_crispra" --bootstrap 2000 --seed 1729
else
  "${PY}" scripts/evaluate_pancreas_crossfit_biology.py \
    --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
    --corrupted "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
    --coordinates "${PAN}/data/pancreas_islets/coordinates/mask_010.parquet" \
    --crossfit-dir "${CF}" --safe-fusion-subdir safe_fusion_b32 \
    --extra-method logistic_2pct=selector_deployment/logistic/safe_fusion_calibrated_0p02 \
    --extra-method logistic_5pct=selector_deployment/logistic/safe_fusion_calibrated_0p05 \
    --extra-method logistic_10pct=selector_deployment/logistic/safe_fusion_calibrated_0p1 \
    --extra-method hist_gbdt_2pct=selector_deployment/hist_gbdt/safe_fusion_calibrated_hist_gbdt_0p02 \
    --extra-method mlp_2pct=selector_deployment/mlp/safe_fusion_calibrated_mlp_0p02 \
    --extra-method mlp_5pct=selector_deployment/mlp/safe_fusion_calibrated_mlp_0p05 \
    --extra-method mlp_10pct=selector_deployment/mlp/safe_fusion_calibrated_mlp_0p1 \
    --extra-method rank_ensemble_2pct=selector_deployment/rank_ensemble/safe_fusion_calibrated_rank_ensemble_0p02 \
    --output-dir "${OUT}/biology/pancreas" --bootstrap 2000 --seed 1729
fi
