#!/usr/bin/env bash
#SBATCH --job-name=sf-thin-transfer
#SBATCH --account=torch_pr_634_general
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-thin-transfer-%x-%A_%a.out
#SBATCH --error=logs/slurm-thin-transfer-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | prepare | cpu | gpu | selector | stacked | evaluate}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
source scripts/unit_paths.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

R=artifacts/paper_evidence/review_round2/thinning_transfer
T=artifacts/paper_evidence/thinning
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets
CF=artifacts/paper_evidence/pancreas_crossfit
NORMAN=artifacts/paper_evidence/review_round2/leakage_free/norman_crispra
TUNITS=(colon_thinning_050 colon_thinning_025 pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2 norman_thinning_050)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)
STACKED=(scvi:scVI magic:MAGIC)
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

unit_paths() {
  local key="$1"
  case "${key}" in
    colon_thinning_*)
      thinned_dir="${T}/data/${key}"; splits="${COL}/splits.parquet"; truth="${COL}/preprocessed.h5ad"
      unit_column=donor; fit=(--fit-split validation --fit-cells 3852)
      thin_root="${T}/${key}"; comparators="${T}/comparators/${key}"
      ;;
    pancreas_thinning_050_*)
      thinned_dir="${T}/data/pancreas_thinning_050"; splits="${CF}/fold_${key##*_}/splits.parquet"
      truth="${PAN}/preprocessed.h5ad"; unit_column=donor; fit=(--fit-split development)
      thin_root="${T}/${key}"; comparators="${T}/comparators/pancreas_thinning_050"
      ;;
    norman_thinning_050)
      thinned_dir="${R}/data/${key}"; splits="${NORMAN}/splits.parquet"
      truth="${NORMAN}/prepared.h5ad"; unit_column=target; fit=(--fit-split development)
      thin_root="${R}/thinning_trained/${key}"; comparators="${R}/thinning_trained/comparators/${key}"
      ;;
    *) echo "unknown unit ${key}" >&2; return 2 ;;
  esac
  mask_root="${R}/mask_trained/${key}"
  hybrid="${mask_root}/input/hybrid.h5ad"
  hybrid_coordinates="${mask_root}/input/coordinates.parquet"
  hybrid_splits="${mask_root}/input/splits.parquet"
}

fit_value() {
  local root="$1" input="$2" coordinates="$3" model="$4" teacher_args=() name
  for name in "${TEACHERS[@]}"; do teacher_args+=(--teacher-contract "${root}/${name}"); done
  .venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion \
    --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
    --output "${root}/$(fused_value_contract "${model}")" --value-model "${model}" \
    --seed 1729 "${teacher_args[@]}"
}

fit_selector() {
  local root="$1" input="$2" coordinates="$3" teacher_args=() name
  fit_value "${root}" "${input}" "${coordinates}" boosted
  fit_value "${root}" "${input}" "${coordinates}" linear
  for name in "${TEACHERS[@]}"; do teacher_args+=(--teacher-contract "${root}/${name}"); done
  .venv/bin/python scripts/calibrated_selective_fill.py \
    --corrupted "${input}" --truth "${truth}" --coordinates "${coordinates}" --splits "${splits}" \
    --fusion-contract "${root}/safe_fusion" "${teacher_args[@]}" \
    --output-dir "${root}/selector" "${fit[@]}" \
    --architecture mlp --budget-mode apply_topk --budgets "${BUDGETS[@]}" \
    --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729 > /dev/null
}

cpu_teacher() {
  local method="$1" root="$2" input="$3"
  if [[ "${method}" == magic_inductive ]]; then
    .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${root}/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
  else
    .venv/bin/python scripts/run_leakage_safe_method.py --method "${method}" \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${root}/${method}" --seed 1729
  fi
}

case "${stage}" in
  submit)
    mkdir -p logs
    prep=$(sbatch --parsable --job-name=sf-tt-prep --array=0-5 --mem=24G --time=00:45:00 "${SCRIPT}" prepare)
    cpu=$(sbatch --parsable --job-name=sf-tt-cpu --array=0-34 --dependency=afterok:${prep} "${SCRIPT}" cpu)
    gpu=$(sbatch --parsable --job-name=sf-tt-gpu --array=0-13 --gres=gpu:l40s:1 --dependency=afterok:${prep} "${SCRIPT}" gpu)
    sel=$(sbatch --parsable --job-name=sf-tt-sel --array=0-6 --mem=48G --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" selector)
    stk=$(sbatch --parsable --job-name=sf-tt-stk --array=0-23 --mem=48G --time=01:30:00 --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" stacked)
    eval=$(sbatch --parsable --job-name=sf-tt-eval --cpus-per-task=4 --mem=48G --time=01:00:00 --dependency=afterok:${sel}:${stk} "${SCRIPT}" evaluate)
    echo "prepare=${prep} cpu=${cpu} gpu=${gpu} selector=${sel} stacked=${stk} evaluate=${eval}"
    ;;
  prepare)
    key="${TUNITS[SLURM_ARRAY_TASK_ID]}"
    unit_paths "${key}"
    if [[ "${key}" == norman_thinning_050 ]]; then
      .venv/bin/python scripts/prepare_thinning_transfer.py thin --truth "${truth}" --splits "${splits}" \
        --unit-column "${unit_column}" --retained-fraction 0.5 --seed 1729 --output-dir "${thinned_dir}"
    fi
    .venv/bin/python scripts/prepare_thinning_transfer.py hybrid --thinned-dir "${thinned_dir}" \
      --splits "${splits}" --unit-column "${unit_column}" --mask-seed 1729 --output-dir "${mask_root}/input"
    ;;
  cpu)
    id="${SLURM_ARRAY_TASK_ID}"
    if (( id < 24 )); then
      unit_paths "${TUNITS[id / 4]}"; coordinates="${hybrid_coordinates}"
      cpu_teacher "${CPU_TEACHERS[id % 4]}" "${mask_root}" "${hybrid}"
    elif (( id < 30 )); then
      unit_paths "${TUNITS[id - 24]}"
      .conda-magic-current/bin/python scripts/run_magic_baseline.py \
        --corrupted "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${mask_root}/magic" --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    elif (( id < 34 )); then
      unit_paths norman_thinning_050; coordinates="${thinned_dir}/coordinates.parquet"
      cpu_teacher "${CPU_TEACHERS[id - 30]}" "${thin_root}" "${thinned_dir}/corrupted.h5ad"
    else
      unit_paths norman_thinning_050
      .conda-magic-current/bin/python scripts/run_magic_baseline.py \
        --corrupted "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet" \
        --splits "${splits}" --output "${comparators}/magic" --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    fi
    echo "elapsed_s=${SECONDS}"
    ;;
  gpu)
    id="${SLURM_ARRAY_TASK_ID}"
    if (( id < 6 )); then
      unit_paths "${TUNITS[id]}"
      .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
        --input "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${mask_root}/scvi_inductive" --seed 1729
    elif (( id < 12 )); then
      unit_paths "${TUNITS[id - 6]}"
      .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
        --corrupted "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${mask_root}/scvi" --epochs 200 --seed 1729
    elif (( id == 12 )); then
      unit_paths norman_thinning_050
      .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
        --input "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet" \
        --splits "${splits}" --output "${thin_root}/scvi_inductive" --seed 1729
    else
      unit_paths norman_thinning_050
      .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
        --corrupted "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet" \
        --splits "${splits}" --output "${comparators}/scvi" --epochs 200 --seed 1729
    fi
    echo "elapsed_s=${SECONDS}"
    ;;
  selector)
    id="${SLURM_ARRAY_TASK_ID}"
    if (( id < 6 )); then
      unit_paths "${TUNITS[id]}"
      fit_selector "${mask_root}" "${hybrid}" "${hybrid_coordinates}"
    else
      unit_paths norman_thinning_050
      fit_selector "${thin_root}" "${thinned_dir}/corrupted.h5ad" "${thinned_dir}/coordinates.parquet"
    fi
    echo "elapsed_s=${SECONDS}"
    ;;
  stacked)
    id="${SLURM_ARRAY_TASK_ID}"
    unit_paths "${TUNITS[(id % 12) / 2]}"
    spec="${STACKED[id % 2]}"; contract="${spec%%:*}"; name="${spec##*:}"
    if (( id < 12 )); then
      input=(--corrupted "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet"
             --contract "${comparators}/${contract}" --output-dir "${R}/thinning_trained/stacked/${TUNITS[(id % 12) / 2]}/${contract}")
    else
      input=(--corrupted "${hybrid}" --coordinates "${hybrid_coordinates}"
             --contract "${mask_root}/${contract}" --output-dir "${mask_root}/stacked/${contract}")
    fi
    .venv/bin/python scripts/stacked_selector_scores.py "${input[@]}" --splits "${splits}" \
      --fit-split "${fit[1]}" --unit-column "${unit_column}" --name "${name}" --seed 1729
    echo "elapsed_s=${SECONDS}"
    ;;
  evaluate)
    .venv/bin/python scripts/evaluate_thinning_transfer.py --root "${R}" --thinning-root "${T}" --bootstrap 2000 --seed 7
    echo "elapsed_s=${SECONDS}"
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
