#!/usr/bin/env bash
#SBATCH --job-name=sf-kd-null
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-kd-null-%A_%a.out
#SBATCH --error=logs/slurm-kd-null-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

SCREEN="${SCREEN:-adamson_crispri}"
SEEDS=(1729 1730 1731 1732)
deployment_paths "${SCREEN}"
OUT_ROOT="${OUT_ROOT:-artifacts/paper_evidence/review_round2/knockdown/pseudolabel_null/${SCREEN}}"
seed="${SEEDS[SLURM_ARRAY_TASK_ID]}"
work="${OUT_ROOT}/seed_${seed}"
input="${work}/hybrid.h5ad"
common=(--coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet")
teachers=("${deploy_methods}/gene_median" "${deploy_methods}/svd_impute" "${work}/methods/graph_smooth_condition"
          "${deploy_methods}/magic_inductive" "${work}/methods/scvi_inductive_condition")
teacher_args=()
for path in "${teachers[@]}"; do teacher_args+=(--teacher-contract "${path}"); done

case "${STAGE:?STAGE must be prepare, knn, scvi, select or evaluate}" in
  prepare)
    .venv/bin/python scripts/knockdown_pseudolabel_null.py prepare --input "${out}/hybrid.h5ad" \
      --splits "${out}/splits.parquet" --output-dir "${work}" --seed "${seed}"
    ;;
  knn)
    .venv/bin/python scripts/run_leakage_safe_method.py --method graph_smooth --condition-column pseudo_target \
      --input "${input}" "${common[@]}" --output "${work}/methods/graph_smooth_condition" --seed 1729
    ;;
  scvi)
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi --condition-column pseudo_target \
      --input "${input}" "${common[@]}" --output "${work}/methods/scvi_inductive_condition" --seed 1729
    ;;
  select)
    .venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" "${common[@]}" --output "${work}/methods/safe_fusion_condition" --seed 1729 "${teacher_args[@]}"
    .venv/bin/python scripts/calibrated_selective_fill.py \
      --corrupted "${input}" --truth "${truth}" "${common[@]}" \
      --fusion-contract "${work}/methods/safe_fusion_condition" "${teacher_args[@]}" \
      --output-dir "${work}/selector_condition" "${fit[@]}" --condition-column pseudo_target \
      --architecture mlp --budget-mode apply_topk --budgets 0.01 0.05 0.1 --curve-points 2 --seed 1729
    ;;
  evaluate)
    dirs=()
    for s in "${SEEDS[@]}"; do dirs+=("${OUT_ROOT}/seed_${s}"); done
    .venv/bin/python scripts/knockdown_pseudolabel_null.py evaluate --recorded "${out}/recorded.h5ad" \
      --splits "${out}/splits.parquet" --label-free-selector "${out}/selector" \
      --true-label-selector "${out}/selector_condition" --pseudo-dirs "${dirs[@]}" \
      --output-dir "${OUT_ROOT}/evaluation" --fdr 0.05 --bootstrap-seed 1729
    ;;
esac
