#!/usr/bin/env bash
#SBATCH --job-name=sf-down-selectors
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7%4
#SBATCH --output=logs/slurm-complete-downstream-selectors-%A_%a.out
#SBATCH --error=logs/slurm-complete-downstream-selectors-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
CF=artifacts/paper_evidence/pancreas_crossfit
NORMAN=artifacts/paper_evidence/norman_crispra
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
keys=(pancreas_0 pancreas_1 pancreas_2 norman_crispra adamson_crispri dixit_ko papalexi_eccite zebrafish)
key="${keys[SLURM_ARRAY_TASK_ID]}"

case "${key}" in
  pancreas_*)
    fold="${key##*_}"
    corrupted="${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad"
    truth="${PAN}/data/pancreas_islets/preprocessed.h5ad"
    coordinates="${PAN}/data/pancreas_islets/coordinates/mask_010.parquet"
    splits="${CF}/fold_${fold}/splits.parquet"
    fusion="${CF}/fold_${fold}/safe_fusion_b32"
    gene_median="${CF}/fold_${fold}/gene_median"
    svd="${CF}/fold_${fold}/svd_impute"
    graph="${CF}/fold_${fold}/graph_smooth"
    output="${CF}/fold_${fold}/selector_mlp_biology_range_fullteachers"
    ;;
  norman_crispra)
    root="${NORMAN}"
    corrupted="${root}/corrupted.h5ad"
    truth=external_data/prepared/norman_crispra.h5ad
    coordinates="${root}/coordinates.parquet"
    splits="${root}/splits.parquet"
    fusion="${root}/methods/safe_fusion"
    gene_median="${root}/methods/gene_median"
    svd="${root}/methods/svd_impute"
    graph="${root}/methods/graph_smooth"
    output=artifacts/paper_evidence/selector_mlp_biology_range_fullteachers/norman_crispra
    ;;
  adamson_crispri|dixit_ko|papalexi_eccite)
    root="artifacts/external_perturbseq/${key}"
    corrupted="${root}/corrupted.h5ad"
    truth="external_data/prepared/${key}.h5ad"
    coordinates="${root}/coordinates.parquet"
    splits="${root}/splits.parquet"
    fusion="${root}/safe_fusion"
    gene_median="${root}/gene_median"
    svd="${root}/svd_impute"
    graph="${root}/graph_smooth"
    output="${root}/selector_mlp_biology_range"
    ;;
  zebrafish)
    root=artifacts/external_trajectory/zebrafish
    corrupted="${root}/corrupted.h5ad"
    truth=external_data/prepared/zebrafish_trajectory.h5ad
    coordinates="${root}/coordinates.parquet"
    splits="${root}/splits.parquet"
    fusion="${root}/safe_fusion"
    gene_median="${root}/gene_median"
    svd="${root}/svd_impute"
    graph="${root}/graph_smooth"
    output="${root}/selector_mlp_biology_range"
    ;;
  *)
    echo "unknown dataset ${key}" >&2
    exit 2
    ;;
esac

"${PY}" scripts/calibrated_selective_fill.py \
  --corrupted "${corrupted}" \
  --truth "${truth}" \
  --coordinates "${coordinates}" \
  --splits "${splits}" \
  --fusion-contract "${fusion}" \
  --teacher-contract "${gene_median}" \
  --teacher-contract "${svd}" \
  --teacher-contract "${graph}" \
  --output-dir "${output}" \
  --fit-split development \
  --architecture mlp \
  --budget-mode apply_topk \
  --budgets "${BUDGETS[@]}" \
  --curve-min-budget 0.001 \
  --curve-max-budget 0.10 \
  --curve-points 100 \
  --seed 1729
