#!/usr/bin/env bash
#SBATCH --job-name=sf-ccf-thin
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-colon-crossfit-thin-%x-%A_%a.out
#SBATCH --error=logs/slurm-colon-crossfit-thin-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | prepare | cpu | gpu | selector | stacked | evaluate}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
source scripts/unit_paths.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

CF="${COLON_CROSSFIT_ROOT:-artifacts/paper_evidence/review_round3/colon_crossfit}"
T="${CF}/thinning"
TUNITS=(colon_thinning_050_0 colon_thinning_050_1 colon_thinning_050_2 colon_thinning_025_0 colon_thinning_025_1 colon_thinning_025_2)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)
STACKED=(scvi:scVI magic:MAGIC)
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
task="${SLURM_ARRAY_TASK_ID:-0}"

unit_paths() {
  local key="$1" level="${1%_*}" fold="${1##*_}"
  case "${level}" in
    colon_thinning_050) retained=0.5 ;;
    colon_thinning_025) retained=0.25 ;;
    *) echo "unknown unit ${key}" >&2; return 2 ;;
  esac
  splits="${CF}/fold_${fold}/splits.parquet"
  truth="${CF}/fold_${fold}/prepared.h5ad"
  fit_cells=$(.venv/bin/python -c "import pandas as pd; print(int(pd.read_parquet('${splits}')['split'].isin(['development', 'validation']).sum()))")
  fit=(--fit-split validation --fit-cells "${fit_cells}")
  thinned_dir="${T}/data/${key}"
  thin_root="${T}/thinning_trained/${key}"
  comparators="${T}/thinning_trained/comparators/${key}"
  thin_stacked="${T}/thinning_trained/stacked/${key}"
  mask_root="${T}/mask_trained/${key}"
  hybrid="${mask_root}/input/hybrid.h5ad"
  hybrid_coordinates="${mask_root}/input/coordinates.parquet"
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
  local method="$1" root="$2" input="$3" coordinates="$4"
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
    prep=$(sbatch --parsable --job-name=sf-cct-prep --array=0-5 --mem=16G --time=00:30:00 "${SCRIPT}" prepare)
    cpu=$(sbatch --parsable --job-name=sf-cct-cpu --array=0-59 --time=01:00:00 --dependency=afterok:${prep} "${SCRIPT}" cpu)
    gpu=$(sbatch --parsable --job-name=sf-cct-gpu --array=0-23 --gres=gpu:l40s:1 --dependency=afterok:${prep} "${SCRIPT}" gpu)
    sel=$(sbatch --parsable --job-name=sf-cct-sel --array=0-11 --partition=cs --mem=48G --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" selector)
    stk=$(sbatch --parsable --job-name=sf-cct-stk --array=0-23 --partition=cs --mem=48G --time=01:30:00 --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" stacked)
    eval=$(sbatch --parsable --job-name=sf-cct-eval --cpus-per-task=4 --mem=48G --time=01:00:00 --dependency=afterok:${sel}:${stk} "${SCRIPT}" evaluate)
    echo "prepare=${prep} cpu=${cpu} gpu=${gpu} selector=${sel} stacked=${stk} evaluate=${eval}"
    ;;
  prepare)
    unit_paths "${TUNITS[task]}"
    .venv/bin/python scripts/prepare_thinning_transfer.py thin --truth "${truth}" --splits "${splits}" \
      --unit-column donor --retained-fraction "${retained}" --seed 1729 --output-dir "${thinned_dir}"
    .venv/bin/python scripts/prepare_thinning_transfer.py hybrid --thinned-dir "${thinned_dir}" \
      --splits "${splits}" --unit-column donor --mask-seed 1729 --output-dir "${mask_root}/input"
    ;;
  cpu)

    if (( task < 24 )); then
      unit_paths "${TUNITS[task / 4]}"
      cpu_teacher "${CPU_TEACHERS[task % 4]}" "${thin_root}" "${thinned_dir}/corrupted.h5ad" "${thinned_dir}/coordinates.parquet"
    elif (( task < 48 )); then
      unit_paths "${TUNITS[(task - 24) / 4]}"
      cpu_teacher "${CPU_TEACHERS[task % 4]}" "${mask_root}" "${hybrid}" "${hybrid_coordinates}"
    elif (( task < 54 )); then
      unit_paths "${TUNITS[task - 48]}"
      .conda-magic-current/bin/python scripts/run_magic_baseline.py \
        --corrupted "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet" \
        --splits "${splits}" --output "${comparators}/magic" --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    else
      unit_paths "${TUNITS[task - 54]}"
      .conda-magic-current/bin/python scripts/run_magic_baseline.py \
        --corrupted "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${mask_root}/magic" --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    fi
    ;;
  gpu)

    unit_paths "${TUNITS[task % 6]}"
    if (( task < 6 )); then
      .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
        --input "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet" \
        --splits "${splits}" --output "${thin_root}/scvi_inductive" --seed 1729
    elif (( task < 12 )); then
      .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
        --corrupted "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet" \
        --splits "${splits}" --output "${comparators}/scvi" --epochs 200 --seed 1729
    elif (( task < 18 )); then
      .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
        --input "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${mask_root}/scvi_inductive" --seed 1729
    else
      .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
        --corrupted "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${mask_root}/scvi" --epochs 200 --seed 1729
    fi
    ;;
  selector)

    unit_paths "${TUNITS[task % 6]}"
    if (( task < 6 )); then
      fit_selector "${thin_root}" "${thinned_dir}/corrupted.h5ad" "${thinned_dir}/coordinates.parquet"
    else
      fit_selector "${mask_root}" "${hybrid}" "${hybrid_coordinates}"
    fi
    ;;
  stacked)

    unit_paths "${TUNITS[(task % 12) / 2]}"
    spec="${STACKED[task % 2]}"; contract="${spec%%:*}"; name="${spec##*:}"
    if (( task < 12 )); then
      input=(--corrupted "${thinned_dir}/corrupted.h5ad" --coordinates "${thinned_dir}/coordinates.parquet"
             --contract "${comparators}/${contract}" --output-dir "${thin_stacked}/${contract}")
    else
      input=(--corrupted "${hybrid}" --coordinates "${hybrid_coordinates}"
             --contract "${mask_root}/${contract}" --output-dir "${mask_root}/stacked/${contract}")
    fi
    .venv/bin/python scripts/stacked_selector_scores.py "${input[@]}" --splits "${splits}" \
      --fit-split validation --unit-column donor --name "${name}" --seed 1729
    ;;
  evaluate)
    .venv/bin/python scripts/evaluate_colon_crossfit_thinning.py --root "${CF}" --bootstrap 2000 --seed 7
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
echo "elapsed_s=${SECONDS}"
