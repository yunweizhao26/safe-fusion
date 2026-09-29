#!/usr/bin/env bash
#SBATCH --job-name=sf-fill-fraction
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-7
#SBATCH --output=logs/slurm-apply-fill-fraction-%A_%a.out
#SBATCH --error=logs/slurm-apply-fill-fraction-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
CF=artifacts/paper_evidence/pancreas_crossfit
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
COL_METHODS="${COL}/methods/standardized/colon_epithelial/mask_010"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    corrupted="${COL}/data/colon_epithelial/corrupted/mask_010.h5ad"
    splits="${COL}/data/colon_epithelial/splits.parquet"
    svd="${COL_METHODS}/svd_impute"
    knn="${COL_METHODS}/graph_smooth"
    out=artifacts/paper_evidence/matched_fraction/colon
    ;;
  1|2|3)
    fold=$((SLURM_ARRAY_TASK_ID - 1))
    corrupted="${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad"
    splits="${CF}/fold_${fold}/splits.parquet"
    svd="${CF}/fold_${fold}/svd_impute"
    knn="${CF}/fold_${fold}/graph_smooth"
    out="${CF}/fold_${fold}/matched_fraction"
    ;;
  4)
    root=artifacts/external_trajectory/zebrafish
    corrupted="${root}/corrupted.h5ad"
    splits="${root}/splits.parquet"
    svd="${root}/svd_impute"
    knn="${root}/graph_smooth"
    out="${root}/matched_fraction"
    ;;
  5|6|7)
    datasets=(unused unused unused unused unused adamson_crispri dixit_ko papalexi_eccite)
    root="artifacts/external_perturbseq/${datasets[SLURM_ARRAY_TASK_ID]}"
    corrupted="${root}/corrupted.h5ad"
    splits="${root}/splits.parquet"
    svd="${root}/svd_impute"
    knn="${root}/graph_smooth"
    out="${root}/matched_fraction"
    ;;
esac

"${PY}" scripts/apply_fill_fraction.py --corrupted "${corrupted}" --splits "${splits}" \
  --method-contract "${svd}" --method-name svd --output-root "${out}"
"${PY}" scripts/apply_fill_fraction.py --corrupted "${corrupted}" --splits "${splits}" \
  --method-contract "${knn}" --method-name weighted_knn --output-root "${out}"
