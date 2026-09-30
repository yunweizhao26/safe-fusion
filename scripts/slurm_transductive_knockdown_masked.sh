#!/usr/bin/env bash
#SBATCH --job-name=sf-r4-kd-masked
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-r4-kd-masked-%A_%a.out
#SBATCH --error=logs/slurm-r4-kd-masked-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUT1=artifacts/paper_evidence/review_round4/transductive_references/knockdown
screens=(norman_crispra adamson_crispri dixit_ko papalexi_eccite)
screen="${screens[SLURM_ARRAY_TASK_ID]}"
ROOT="${OUT1}/masked/${screen}"
if [[ "${screen}" == norman_crispra ]]; then
  truth="${ROOT}/prepared.h5ad"
else
  truth="external_data/prepared/${screen}.h5ad"
fi
PY=.venv/bin/python
common=(--input "${ROOT}/corrupted.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" --seed 1729)

case "${STAGE:?STAGE must be teachers, magic, scvi, scvi_condition, stack or select}" in
  teachers)
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" --transductive "${common[@]}" --output "${ROOT}/${method}"
    done
    "${PY}" scripts/run_leakage_safe_method.py --method graph_smooth --transductive --condition-column target \
      "${common[@]}" --output "${ROOT}/graph_smooth_condition"
    ;;
  magic)
    .conda-magic-current/bin/python scripts/run_magic_baseline.py \
      --corrupted "${ROOT}/corrupted.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --output "${ROOT}/magic_standard" --n-jobs "${SLURM_CPUS_PER_TASK}" --seed 1729
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/magic_standard" --corrupted "${ROOT}/corrupted.h5ad" \
      --output "${ROOT}/magic_inductive"
    ;;
  scvi)
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${ROOT}/corrupted.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --output "${ROOT}/scvi_standard" --epochs 200 --seed 1729
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/scvi_standard" --corrupted "${ROOT}/corrupted.h5ad" \
      --output "${ROOT}/scvi_inductive"
    ;;
  scvi_condition)
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${ROOT}/corrupted.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --output "${ROOT}/scvi_standard_condition" --epochs 200 --seed 1729 --condition-column target
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/scvi_standard_condition" --corrupted "${ROOT}/corrupted.h5ad" \
      --output "${ROOT}/scvi_inductive_condition"
    ;;
  stack)
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${ROOT}/safe_fusion" \
      --teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth" --teacher-contract "${ROOT}/magic_inductive" \
      --teacher-contract "${ROOT}/scvi_inductive"
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${ROOT}/safe_fusion_condition" \
      --teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth_condition" --teacher-contract "${ROOT}/magic_inductive" \
      --teacher-contract "${ROOT}/scvi_inductive_condition"
    ;;
  select)
    plain=(--teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth" --teacher-contract "${ROOT}/magic_inductive" \
      --teacher-contract "${ROOT}/scvi_inductive")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/corrupted.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/safe_fusion" "${plain[@]}" \
      --output-dir "${ROOT}/selector_mlp_biology_range" --fit-split development --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729
    labelled=(--teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth_condition" --teacher-contract "${ROOT}/magic_inductive" \
      --teacher-contract "${ROOT}/scvi_inductive_condition")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/corrupted.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/safe_fusion_condition" "${labelled[@]}" \
      --output-dir "${ROOT}/selector_condition" --fit-split development --condition-column target --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
