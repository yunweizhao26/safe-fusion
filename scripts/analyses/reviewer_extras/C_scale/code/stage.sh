#!/usr/bin/env bash
#SBATCH --job-name=scale
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-scale-%x-%A_%a.out
#SBATCH --error=logs/slurm-scale-%x-%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

PY=.venv/bin/python
ROOT="${SCALE_ROOT:?SCALE_ROOT must point inside reviewer_extras/C_scale}"
SEED=1729
SIZES=(200000)
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
TRANSDUCTIVE_TEACHERS=(gene_median svd_impute graph_smooth magic scvi)
WITHOUT_MAGIC=(gene_median svd_impute graph_smooth scvi_inductive)
GPU_STAGES=(scvi_teacher scvi)
declare -A MEM_GB=(
  [mask]="32 32 48 96" [teachers]="32 48 96 192" [scvi_teacher]="32 48 96 160" [stack]="32 48 96 160"
  [selector]="32 48 96 160" [magic]="32 64 128 256" [scvi]="32 48 96 160" [scvi_probability]="32 48 96 160"
  [alra]="32 48 96 160" [saver]="64 96 192 256" [transductive_teachers]="32 48 96 160"
  [transductive_stack]="32 48 96 160" [transductive_selector]="32 48 96 160"
  [without_magic_stack]="32 48 96 160" [without_magic_selector]="32 48 96 160"
)
declare -A HOURS=(
  [mask]="6 6 6 6" [teachers]="12 12 24 24" [scvi_teacher]="12 24 24 24" [stack]="12 12 24 24"
  [selector]="12 12 24 24" [magic]="12 12 24 24" [scvi]="12 12 24 24" [scvi_probability]="6 6 6 6"
  [alra]="12 12 24 24" [saver]="24 24 24 24" [transductive_teachers]="6 6 12 12"
  [transductive_stack]="12 12 24 24" [transductive_selector]="12 12 24 24"
  [without_magic_stack]="12 12 24 24" [without_magic_selector]="12 12 24 24"
)

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
task="${SLURM_ARRAY_TASK_ID:-0}"
step() { echo "step=$1 elapsed_s=${SECONDS}"; }

unit_paths() {
  unit="${ROOT}/cells_${SIZES[$1]}"
  truth="${unit}/truth.h5ad"
  splits="${unit}/splits.parquet"
  input="${unit}/masked/corrupted.h5ad"
  coordinates="${unit}/masked/coordinates.parquet"
}

teacher_args() {
  local name
  for name in gene_median svd_impute graph_smooth magic_inductive scvi_inductive; do
    printf -- '--teacher-contract\n%s\n' "$1/${name}"
  done
}

contract_args() {
  local root="$1" name
  shift
  for name in "$@"; do
    printf -- '--teacher-contract\n%s\n' "${root}/${name}"
  done
}

transductive_args() {
  local name
  for name in "${TRANSDUCTIVE_TEACHERS[@]}"; do
    printf -- '--teacher-contract\n%s\n' "$1/${name}"
  done
}

case "${STAGE}" in
  prepare)
    "${PY}" scripts/scale_prepare.py --sizes "${SIZES[@]}" --variable-genes 5000 --output-dir "${ROOT}" --seed "${SEED}"
    ;;
  mask)
    unit_paths "${task}"
    "${PY}" scripts/make_mask_replicate.py --truth "${truth}" --splits "${splits}" \
      --unit-column donor --seed "${SEED}" --output-dir "${unit}/masked"
    ;;
  teachers)
    unit_paths $((task / 4))
    method="${TEACHERS[task % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${unit}/methods/${method}" --seed "${SEED}" --n-jobs "${SLURM_CPUS_PER_TASK}" \
        --extension-block-cells 1024
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${unit}/methods/${method}" --seed "${SEED}"
    fi
    ;;
  scvi_teacher)
    unit_paths "${task}"
    .conda-scvi-current/bin/python "${ROOT}/code/run_scvi.py" scripts/run_inductive_teacher.py --method scvi \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${unit}/methods/scvi_inductive" --seed "${SEED}"
    ;;
  stack)
    unit_paths "${task}"
    mapfile -t teachers < <(teacher_args "${unit}/methods")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${unit}/methods/safe_fusion" --seed "${SEED}" "${teachers[@]}"
    ;;
  selector)
    unit_paths "${task}"
    mapfile -t teachers < <(teacher_args "${unit}/methods")
    "${PY}" scripts/scale_selector.py --corrupted "${input}" --coordinates "${coordinates}" \
      --splits "${splits}" --fusion-contract "${unit}/methods/safe_fusion" "${teachers[@]}" \
      --fit-split development --output-dir "${unit}/selector" --seed "${SEED}"
    ;;
  transductive_teachers)
    unit_paths $((task / 5))
    method="${TRANSDUCTIVE_TEACHERS[task % 5]}"
    if [[ "${method}" == magic || "${method}" == scvi ]]; then
      "${PY}" scripts/count_scale_contract.py --contract "${unit}/baselines/${method}" \
        --corrupted "${input}" --output "${unit}/transductive/${method}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" --transductive \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${unit}/transductive/${method}" --seed "${SEED}"
    fi
    ;;
  transductive_stack)
    unit_paths "${task}"
    mapfile -t teachers < <(transductive_args "${unit}/transductive")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${unit}/transductive/safe_fusion" --seed "${SEED}" "${teachers[@]}"
    ;;
  transductive_selector)
    unit_paths "${task}"
    mapfile -t teachers < <(transductive_args "${unit}/transductive")
    "${PY}" scripts/scale_selector.py --corrupted "${input}" --coordinates "${coordinates}" \
      --splits "${splits}" --fusion-contract "${unit}/transductive/safe_fusion" "${teachers[@]}" \
      --fit-split development --output-dir "${unit}/transductive/selector" --seed "${SEED}"
    ;;
  without_magic_stack)
    unit_paths "${task}"
    mapfile -t teachers < <(contract_args "${unit}/methods" "${WITHOUT_MAGIC[@]}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${unit}/without_magic/safe_fusion" --seed "${SEED}" "${teachers[@]}"
    ;;
  without_magic_selector)
    unit_paths "${task}"
    mapfile -t teachers < <(contract_args "${unit}/methods" "${WITHOUT_MAGIC[@]}")
    "${PY}" scripts/scale_selector.py --corrupted "${input}" --coordinates "${coordinates}" \
      --splits "${splits}" --fusion-contract "${unit}/without_magic/safe_fusion" "${teachers[@]}" \
      --fit-split development --output-dir "${unit}/without_magic/selector" --seed "${SEED}"
    ;;
  stack_replicate)
    unit_paths $((task % 4))
    if (( task < 4 )); then
      mapfile -t teachers < <(teacher_args "${unit}/methods")
      "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${unit}/timing_replicate/safe_fusion" --seed "${SEED}" "${teachers[@]}"
    elif (( task < 8 )); then
      mapfile -t teachers < <(transductive_args "${unit}/transductive")
      "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${unit}/timing_replicate/transductive_safe_fusion" --seed "${SEED}" "${teachers[@]}"
    else
      mapfile -t teachers < <(contract_args "${unit}/methods" "${WITHOUT_MAGIC[@]}")
      "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${unit}/timing_replicate/without_magic_safe_fusion" --seed "${SEED}" "${teachers[@]}"
    fi
    ;;
  check)

    LF=artifacts/paper_evidence/review_round2/leakage_free
    names=(pancreas_0 norman_crispra)
    data=("${LF}/pancreas_crossfit/fold_0" "${LF}/norman_crispra")
    methods=("${LF}/pancreas_crossfit/fold_0" "${LF}/norman_crispra/methods")
    mapfile -t teachers < <(teacher_args "${methods[task]}")
    "${PY}" scripts/scale_selector.py --corrupted "${data[task]}/corrupted.h5ad" \
      --coordinates "${data[task]}/coordinates.parquet" --splits "${data[task]}/splits.parquet" \
      --fusion-contract "${methods[task]}/safe_fusion" "${teachers[@]}" --fit-split development \
      --output-dir "${ROOT}/checks/selector_reproduction/${names[task]}" --seed "${SEED}"
    ;;
  magic)
    unit_paths "${task}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${unit}/baselines/magic" \
      --n-jobs "${SLURM_CPUS_PER_TASK}" --seed "${SEED}"
    ;;
  scvi)
    unit_paths "${task}"
    .conda-scvi-current/bin/python "${ROOT}/code/run_scvi.py" scripts/run_scvi_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${unit}/baselines/scvi" \
      --epochs 200 --seed "${SEED}"
    ;;
  scvi_probability)
    unit_paths "${task}"
    .conda-scvi-current/bin/python "${ROOT}/code/run_scvi.py" scripts/nonzero_probability.py --method scvi \
      --contract "${unit}/baselines/scvi" --corrupted "${input}" --output "${unit}/baselines/scvi_probability"
    ;;
  alra)
    unit_paths "${task}"
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    "${PY}" scripts/run_alra_baseline.py --corrupted "${input}" --splits "${splits}" \
      --output "${unit}/baselines/alra" --seed "${SEED}"
    ;;
  saver)
    unit_paths "${task}"
    "${PY}" scripts/run_saver_baseline.py --corrupted "${input}" --splits "${splits}" \
      --output "${unit}/baselines/saver" --ncores "${SLURM_CPUS_PER_TASK}"
    ;;
  evaluate)
    "${PY}" "${ROOT}/code/scale_evaluate.py" --root "${ROOT}" --sizes "${SIZES[@]}" --seed "${SEED}"
    if [[ -f "${ROOT}/cells_200000/selector/test_score.npy" ]]; then
    "${PY}" scripts/paired_masked_f1_bootstrap.py --unit-counts "${ROOT}/results/masked_f1_unit_counts.parquet" \
      --stacked-root "${ROOT}/results/no_stacked_selectors" --output "${ROOT}/results/masked_f1_paired_bootstrap.csv" \
      --summary-output "${ROOT}/results/masked_f1_paired_bootstrap.json" --draws 2000 --seed "${SEED}"
    fi
    ;;
  report)
    "${PY}" scripts/scale_report.py --root "${ROOT}"
    "${PY}" scripts/scale_plot.py --root "${ROOT}"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
step "${STAGE}"
