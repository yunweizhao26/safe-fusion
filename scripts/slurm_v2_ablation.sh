#!/usr/bin/env bash
#SBATCH --job-name=sf-v2-ablation
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-v2-ablation-%x-%A_%a.out
#SBATCH --error=logs/slurm-v2-ablation-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

PY=.venv/bin/python
ABL=artifacts/paper_evidence/review_round3/value_v2_ablations/ablations
UNITS=(pancreas_0 pancreas_1 pancreas_2 colon norman_crispra colon_0 colon_1 colon_2)
RATES=(0.05 0.20)
RATE_NAMES=(rho_0p05 rho_0p20)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)
NOCF_TEACHERS=(svd_impute graph_smooth magic_inductive)
HIDDEN=("64 32" "128 64 32" "256 128 64")
HIDDEN_NAMES=(h64_32 h128_64_32 h256_128_64)
LEAVES=(31 31 31 63 63 63)
MINLEAF=(50 100 200 50 100 200)
STAGE="${STAGE:?set STAGE}"
task="${SLURM_ARRAY_TASK_ID:-0}"
step() { echo "step=$1 elapsed_s=${SECONDS}"; }

unit_paths() {
  read -r corrupted coordinates splits truth teacher_root fit_split unit_column < <(
    "${PY}" scripts/v2_ablation_common.py --unit "$1" --unit-paths)
}

mask_pair() {
  key="${UNITS[$1 / 2]}"
  rate="${RATES[$1 % 2]}"
  root="${ABL}/mask_rate/${RATE_NAMES[$1 % 2]}/${key}"
  input="${root}/input/corrupted.h5ad"
  input_coordinates="${root}/input/coordinates.parquet"
  unit_paths "${key}"
}

teacher_args() {
  local directory="$1" name
  for name in "${TEACHERS[@]}"; do
    if [[ "${name}" == gene_median && -n "${2:-}" ]]; then
      printf -- '--teacher-contract\n%s\n' "$2"
    else
      printf -- '--teacher-contract\n%s\n' "${directory}/${name}"
    fi
  done
}

case "${STAGE}" in
  mask_input)
    mask_pair "${task}"
    "${PY}" scripts/v2_ablation_mask_rate.py --unit "${key}" --mask-rate "${rate}" --seed 1729
    ;;
  mask_teachers)
    mask_pair "$((task / 4))"
    method="${CPU_TEACHERS[task % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${input}" --coordinates "${input_coordinates}" --splits "${splits}" \
        --output "${root}/methods/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${input}" --coordinates "${input_coordinates}" --splits "${splits}" \
        --output "${root}/methods/${method}" --seed 1729
    fi
    ;;
  mask_scvi)
    mask_pair "${task}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${input}" --coordinates "${input_coordinates}" --splits "${splits}" \
      --output "${root}/methods/scvi_inductive" --seed 1729
    ;;
  mask_stack)
    mask_pair "${task}"
    mapfile -t teachers < <(teacher_args "${root}/methods")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${input_coordinates}" --splits "${splits}" \
      --output "${root}/methods/safe_fusion" --value-model boosted --seed 1729 "${teachers[@]}"
    ;;
  mask_selector)
    mask_pair "${task}"
    "${PY}" scripts/v2_ablation_selector.py --unit "${key}" --corrupted "${input}" \
      --coordinates "${input_coordinates}" --teacher-root "${root}/methods" \
      --output-dir "${root}/selector" --seed 1729
    ;;
  nocf_teachers)
    key="${UNITS[task / 3]}"
    method="${NOCF_TEACHERS[task % 3]}"
    unit_paths "${key}"
    root="${ABL}/no_crossfit/${key}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/v2_ablation_teachers.py --method magic \
        --input "${corrupted}" --coordinates "${coordinates}" --splits "${splits}" \
        --production-contract "${teacher_root}/${method}" \
        --output "${root}/methods/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
    else
      "${PY}" scripts/v2_ablation_teachers.py --method "${method}" \
        --input "${corrupted}" --coordinates "${coordinates}" --splits "${splits}" \
        --production-contract "${teacher_root}/${method}" \
        --output "${root}/methods/${method}" --seed 1729
    fi
    ;;
  nocf_scvi)
    key="${UNITS[task]}"
    unit_paths "${key}"
    .conda-scvi-current/bin/python scripts/v2_ablation_teachers.py --method scvi \
      --input "${corrupted}" --coordinates "${coordinates}" --splits "${splits}" \
      --production-contract "${teacher_root}/scvi_inductive" \
      --output "${ABL}/no_crossfit/${key}/methods/scvi_inductive" --seed 1729
    ;;
  nocf_stack)
    key="${UNITS[task]}"
    unit_paths "${key}"
    root="${ABL}/no_crossfit/${key}"
    mapfile -t teachers < <(teacher_args "${root}/methods" "${teacher_root}/gene_median")
    "${PY}" scripts/v2_ablation_value.py --input "${corrupted}" --coordinates "${coordinates}" \
      --splits "${splits}" --output "${root}/methods/safe_fusion" --seed 1729 "${teachers[@]}"
    ;;
  nocf_selector)
    key="${UNITS[task]}"
    unit_paths "${key}"
    root="${ABL}/no_crossfit/${key}"
    "${PY}" scripts/v2_ablation_selector.py --unit "${key}" --teacher-root "${root}/methods" \
      --gene-median-contract "${teacher_root}/gene_median" --output-dir "${root}/selector" --seed 1729
    ;;
  selector_size)
    key="${UNITS[task / 3]}"

    "${PY}" scripts/v2_ablation_selector.py --unit "${key}" --hidden ${HIDDEN[task % 3]} \
      --output-dir "${ABL}/selector_size/${HIDDEN_NAMES[task % 3]}/${key}" --seed 1729
    ;;
  value_boosting)
    key="${UNITS[task / 6]}"
    setting=$((task % 6))
    unit_paths "${key}"
    mapfile -t teachers < <(teacher_args "${teacher_root}")
    "${PY}" scripts/v2_ablation_value.py --input "${corrupted}" --coordinates "${coordinates}" \
      --splits "${splits}" --max-leaf-nodes "${LEAVES[setting]}" --min-samples-leaf "${MINLEAF[setting]}" \
      --output "${ABL}/value_boosting/leaves${LEAVES[setting]}_minleaf${MINLEAF[setting]}/${key}" \
      --seed 1729 "${teachers[@]}"
    ;;
  evaluate)
    "${PY}" scripts/v2_ablation_evaluate.py --seed 1729 ${FAMILIES:+--families ${FAMILIES}}
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
step "${STAGE}"
