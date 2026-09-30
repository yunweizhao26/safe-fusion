#!/usr/bin/env bash
#SBATCH --job-name=sf-thin-trans
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-thin-transductive-%x-%A_%a.out
#SBATCH --error=logs/slurm-thin-transductive-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | teachers | stack | selector}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

CCF=artifacts/paper_evidence/review_round3/colon_crossfit
CCF_T="${CCF}/thinning"
PCF=artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit
PCF_T=artifacts/paper_evidence/review_round4/transductive_main/pancreas_rebuilt_thinning
NORMAN=artifacts/paper_evidence/review_round2/leakage_free/norman_crispra
NORMAN_T=artifacts/paper_evidence/review_round2/thinning_transfer
OUT=artifacts/paper_evidence/review_round4/transductive_main/thinning

KEYS=(colon_thinning_050_0 colon_thinning_050_1 colon_thinning_050_2 \
      colon_thinning_025_0 colon_thinning_025_1 colon_thinning_025_2 \
      pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2 \
      norman_thinning_050)
DESIGNS=(thinning-trained mask-trained)

unit_paths() {
  local key="$1" design="$2" fold
  case "${key}" in
    colon_thinning_050_*|colon_thinning_025_*)
      fold="${key##*_}"
      splits="${CCF}/fold_${fold}/splits.parquet"; truth="${CCF}/fold_${fold}/prepared.h5ad"
      unit_column=donor; fit_split=validation
      if [[ "${design}" == thinning-trained ]]; then
        input="${CCF_T}/data/${key}/corrupted.h5ad"; coordinates="${CCF_T}/data/${key}/coordinates.parquet"
        standard_root="${CCF_T}/thinning_trained/comparators/${key}"
      else
        input="${CCF_T}/mask_trained/${key}/input/hybrid.h5ad"; coordinates="${CCF_T}/mask_trained/${key}/input/coordinates.parquet"
        standard_root="${CCF_T}/mask_trained/${key}"
      fi
      ;;
    pancreas_thinning_050_*)
      fold="${key##*_}"
      splits="${PCF}/fold_${fold}/splits.parquet"; truth="${PCF}/fold_${fold}/prepared.h5ad"
      unit_column=donor; fit_split=development
      if [[ "${design}" == thinning-trained ]]; then
        input="${PCF_T}/data/${key}/corrupted.h5ad"; coordinates="${PCF_T}/data/${key}/coordinates.parquet"
        standard_root="${PCF_T}/thinning_trained/comparators/${key}"
      else
        input="${PCF_T}/mask_trained/${key}/input/hybrid.h5ad"; coordinates="${PCF_T}/mask_trained/${key}/input/coordinates.parquet"
        standard_root="${PCF_T}/mask_trained/${key}"
      fi
      ;;
    norman_thinning_050)
      splits="${NORMAN}/splits.parquet"; truth="${NORMAN}/prepared.h5ad"
      unit_column=target; fit_split=development
      if [[ "${design}" == thinning-trained ]]; then
        input="${NORMAN_T}/data/${key}/corrupted.h5ad"; coordinates="${NORMAN_T}/data/${key}/coordinates.parquet"
        standard_root="${NORMAN_T}/thinning_trained/comparators/${key}"
      else
        input="${NORMAN_T}/mask_trained/${key}/input/hybrid.h5ad"; coordinates="${NORMAN_T}/mask_trained/${key}/input/coordinates.parquet"
        standard_root="${NORMAN_T}/mask_trained/${key}"
      fi
      ;;
    *) echo "unknown key ${key}" >&2; return 2 ;;
  esac
  out_root="${OUT}/${key}/${design}"
}

combo_key() { local idx="$1"; echo "${KEYS[idx / 2]}"; }
combo_design() { local idx="$1"; echo "${DESIGNS[idx % 2]}"; }

case "${stage}" in
  submit)
    mkdir -p logs
    t=$(sbatch --parsable --job-name=sf-thtr-teach --array=0-99 "${SCRIPT}" teachers)
    s=$(sbatch --parsable --job-name=sf-thtr-stack --array=0-19 --mem=48G --dependency=afterok:${t} "${SCRIPT}" stack)
    sel=$(sbatch --parsable --job-name=sf-thtr-sel --array=0-19 --mem=48G --time=01:00:00 --dependency=afterok:${s} "${SCRIPT}" selector)
    echo "teachers=${t} stack=${s} selector=${sel}"
    ;;
  teachers)
    teachers=(gene_median svd_impute graph_smooth magic scvi)
    combo=$(( SLURM_ARRAY_TASK_ID / 5 )); teacher="${teachers[SLURM_ARRAY_TASK_ID % 5]}"
    key=$(combo_key "${combo}"); design=$(combo_design "${combo}")
    unit_paths "${key}" "${design}"
    output="${out_root}/${teacher}"
    case "${teacher}" in
      magic|scvi)
        .venv/bin/python scripts/count_scale_contract.py \
          --contract "${standard_root}/${teacher}" --corrupted "${input}" --output "${output}"
        ;;
      *)
        .venv/bin/python scripts/run_leakage_safe_method.py --method "${teacher}" --transductive \
          --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
          --output "${output}" --seed 1729
        ;;
    esac
    ;;
  stack)
    key=$(combo_key "${SLURM_ARRAY_TASK_ID}"); design=$(combo_design "${SLURM_ARRAY_TASK_ID}")
    unit_paths "${key}" "${design}"
    teacher_args=()
    for teacher in gene_median svd_impute graph_smooth magic scvi; do
      teacher_args+=(--teacher-contract "${out_root}/${teacher}")
    done
    .venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion --transductive \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${out_root}/safe_fusion" --seed 1729 "${teacher_args[@]}"
    ;;
  selector)
    key=$(combo_key "${SLURM_ARRAY_TASK_ID}"); design=$(combo_design "${SLURM_ARRAY_TASK_ID}")
    unit_paths "${key}" "${design}"
    teacher_args=()
    for teacher in gene_median svd_impute graph_smooth magic scvi; do
      teacher_args+=(--teacher-contract "${out_root}/${teacher}")
    done
    .venv/bin/python scripts/calibrated_selective_fill.py \
      --corrupted "${input}" --truth "${truth}" --coordinates "${coordinates}" --splits "${splits}" \
      --fusion-contract "${out_root}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${out_root}/selector" --fit-split "${fit_split}" \
      --architecture mlp --budget-mode apply_topk --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729 > /dev/null
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
echo "elapsed_s=${SECONDS}"
