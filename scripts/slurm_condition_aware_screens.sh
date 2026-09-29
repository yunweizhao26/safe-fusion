#!/usr/bin/env bash
#SBATCH --job-name=sf-condition
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-condition-%A_%a.out
#SBATCH --error=logs/slurm-condition-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

screens=(adamson_crispri dixit_ko papalexi_eccite)
settings=(masked deployment)
screen="${screens[SLURM_ARRAY_TASK_ID / 2]}"
setting="${settings[SLURM_ARRAY_TASK_ID % 2]}"
if [[ "${setting}" == masked ]]; then
  unit_paths "${screen}"
  root="${methods_root}"
  selector_output="${methods_root}/selector_condition"
  curve=(--curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000)
else
  deployment_paths "${screen}"
  input="${out}/hybrid.h5ad"
  coordinates="${out}/coordinates.parquet"
  splits="${out}/splits.parquet"
  root="${deploy_methods}"
  selector_output="${out}/selector_condition"
  curve=(--curve-points 2)
fi
teachers=(gene_median svd_impute graph_smooth_condition magic_inductive scvi_inductive_condition)
teacher_args=()
for name in "${teachers[@]}"; do teacher_args+=(--teacher-contract "${root}/${name}"); done

case "${STAGE:?STAGE must be knn, scvi, select or matched}" in
  knn)
    .venv/bin/python scripts/run_leakage_safe_method.py --method graph_smooth --condition-column target \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${root}/graph_smooth_condition" --seed 1729
    ;;
  scvi)
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi --condition-column target \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${root}/scvi_inductive_condition" --seed 1729
    ;;
  select)
    .venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${root}/safe_fusion_condition" --seed 1729 "${teacher_args[@]}"
    .venv/bin/python scripts/calibrated_selective_fill.py \
      --corrupted "${input}" --truth "${truth}" --coordinates "${coordinates}" --splits "${splits}" \
      --fusion-contract "${root}/safe_fusion_condition" "${teacher_args[@]}" \
      --output-dir "${selector_output}" --fit-split development --condition-column target \
      --architecture mlp --budget-mode apply_topk --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      "${curve[@]}" --seed 1729
    ;;
  matched)
    for spec in "graph_smooth_condition knn_condition" "scvi_inductive_condition scvi_condition" \
                "scvi_inductive scvi" "magic_inductive magic"; do
      set -- ${spec}
      .venv/bin/python scripts/apply_fill_fraction.py --corrupted "${input}" --splits "${splits}" \
        --method-contract "${root}/$1" --method-name "$2" --output-root "$(dirname "${selector_output}")/matched_condition" \
        --fractions 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10
    done
    ;;
esac
