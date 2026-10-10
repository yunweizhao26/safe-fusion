#!/usr/bin/env bash
#SBATCH --job-name=sf-v2-value
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-v2-value-%x-%A_%a.out
#SBATCH --error=logs/slurm-v2-value-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
OUT=artifacts/paper_evidence/review_round3/value_v2_ablations/value
LF=artifacts/paper_evidence/review_round2/leakage_free
TT=artifacts/paper_evidence/review_round2/thinning_transfer/mask_trained
NR=artifacts/paper_evidence/review_round2/norman_rebuilt
DEPLOY=artifacts/paper_evidence/downstream_deployment
SLE=artifacts/paper_evidence/sle_treg_case
CCF=artifacts/paper_evidence/review_round3/colon_crossfit
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial
COL_METHODS=artifacts/colon_runs/0b2469810675-c0db6f963e94/methods/standardized/colon_epithelial/mask_010
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets
STAGE="${STAGE:?set STAGE}"
task="${SLURM_ARRAY_TASK_ID:-0}"
PCTS=(1 5 10)
VALUES=(conditional rate_poisson rate_negbin conditional_detection poisson_all poisson_all_unweighted)
ALTERNATIVES=("${VALUES[@]:1}")
FIT_UNITS=(pancreas_0 pancreas_1 pancreas_2 colon norman_crispra
           colon_thinning_050 colon_thinning_025 pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2 norman_thinning_050
           deploy_pancreas_0 deploy_pancreas_1 deploy_pancreas_2 deploy_colon deploy_zebrafish deploy_norman_crispra
           sle_main_0 sle_main_1 sle_main_2
           colon_0 colon_1 colon_2 colon_thinning_050_0 colon_thinning_050_1 colon_thinning_050_2
           colon_thinning_025_0 colon_thinning_025_1 colon_thinning_025_2)
DEPLOY_UNITS=(pancreas_0 pancreas_1 pancreas_2 colon zebrafish norman_crispra)
step() { echo "step=$1 elapsed_s=${SECONDS}"; }

fit_paths() {
  local key="$1" root
  fit_split=development
  case "${key}" in
    pancreas_?)
      root="${LF}/pancreas_crossfit/fold_${key##*_}"
      input="${root}/corrupted.h5ad"; coordinates="${root}/coordinates.parquet"; splits="${root}/splits.parquet"; teachers="${root}"
      ;;
    colon)
      input="${COL}/corrupted/mask_010.h5ad"; coordinates="${COL}/coordinates/mask_010.parquet"; splits="${COL}/splits.parquet"
      teachers="${COL_METHODS}"; fit_split=validation
      ;;
    colon_?)
      root="${CCF}/fold_${key##*_}"
      input="${root}/corrupted.h5ad"; coordinates="${root}/coordinates.parquet"; splits="${root}/splits.parquet"; teachers="${root}"
      fit_split=validation
      ;;
    colon_thinning_*_?)
      root="${CCF}/thinning/mask_trained/${key}"
      input="${root}/input/hybrid.h5ad"; coordinates="${root}/input/coordinates.parquet"; splits="${root}/input/splits.parquet"
      teachers="${root}"; fit_split=validation
      ;;
    norman_crispra)
      root="${LF}/norman_crispra"
      input="${root}/corrupted.h5ad"; coordinates="${root}/coordinates.parquet"; splits="${root}/splits.parquet"; teachers="${root}/methods"
      ;;
    *_thinning_*)
      root="${TT}/${key}"
      input="${root}/input/hybrid.h5ad"; coordinates="${root}/input/coordinates.parquet"; splits="${root}/input/splits.parquet"
      teachers="${root}"
      if [[ "${key}" == colon_* ]]; then fit_split=validation; fi
      ;;
    deploy_*)
      deploy_source "${key#deploy_}"
      input="${source}/hybrid.h5ad"; coordinates="${source}/coordinates.parquet"; splits="${source}/splits.parquet"
      teachers="${source}/methods"
      if [[ "${key}" == deploy_colon ]]; then fit_split=validation; fi
      ;;
    sle_main_?)
      root="${SLE}/main/fold_${key##*_}/deploy"
      input="${root}/hybrid.h5ad"; coordinates="${root}/coordinates.parquet"; splits="${root}/splits.parquet"; teachers="${root}/methods"
      ;;
    *) echo "unknown fit unit ${key}" >&2; return 2 ;;
  esac
}

deploy_source() {
  case "$1" in
    pancreas_*) source="${DEPLOY}/pancreas/fold_${1##*_}"; unit_dir="pancreas/fold_${1##*_}" ;;
    norman_crispra) source="${NR}/deployment/norman_crispra"; unit_dir=norman_crispra ;;
    *) source="${DEPLOY}/$1"; unit_dir="$1" ;;
  esac
}

deploy_methods() {
  local pct value
  deploy_source "$1"
  methods=()
  for pct in "${PCTS[@]}"; do
    methods+=(--method "safe_fusion_${pct}pct=${source}/safe_fusion_${pct}pct")
    for value in "${ALTERNATIVES[@]}"; do
      methods+=(--method "safe_fusion_${value}_${pct}pct=${OUT}/deployment/fills/${unit_dir}/safe_fusion_${value}_${pct}pct")
    done
  done
}

case "${STAGE}" in
  inner)
    "${PY}" scripts/v2_value_inner.py --array-index "${task}" --output-root "${OUT}/inner" --seed 1729
    ;;
  select)

    "${PY}" scripts/v2_value_select.py --inner-root "${OUT}/inner" --output-dir "${OUT}/selection" --draws 2000 --seed 1729
    "${PY}" scripts/v2_value_select.py --inner-root "${OUT}/inner" --output-dir "${OUT}/selection_colon_locked_split" \
      --units pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2 norman_thinning_050 colon_thinning_050 \
        pancreas_0 pancreas_1 pancreas_2 norman_crispra colon --draws 2000 --seed 1729
    ;;
  fit)
    key="${FIT_UNITS[task]}"
    fit_paths "${key}"
    "${PY}" scripts/v2_value_models.py --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --teacher-root "${teachers}" --fit-split "${fit_split}" --values "${VALUES[@]}" \
      --output-root "${OUT}/contracts/${key}" --seed 1729
    ;;
  evaluate)
    "${PY}" scripts/v2_value_evaluate.py --contract-root "${OUT}/contracts" --values "${VALUES[@]}" \
      --output-dir "${OUT}/test" --draws 2000 --seed 1729
    ;;
  deploy_fill)
    deploy_source "${DEPLOY_UNITS[task]}"
    for value in "${ALTERNATIVES[@]}"; do
      "${PY}" scripts/v2_value_deploy.py --unit-dir "${source}" \
        --value-contract "${OUT}/contracts/deploy_${DEPLOY_UNITS[task]}/${value}" --name "safe_fusion_${value}" \
        --pcts "${PCTS[@]}" --output-root "${OUT}/deployment/fills/${unit_dir}"
    done
    ;;
  deploy_eval)
    EVAL="${OUT}/deployment/evaluation"
    case "${task}" in
      0)
        deploy_methods colon
        "${PY}" scripts/evaluate_colon_donor_biology.py --truth "${COL}/preprocessed.h5ad" \
          --corrupted "${source}/recorded.h5ad" --coordinates "${source}/empty_coordinates.parquet" \
          --splits "${source}/splits.parquet" "${methods[@]}" --output-dir "${EVAL}/markers/colon" \
          --bootstrap 2000 --seed 1729 --marker-panel source
        ;;
      1)
        extras=()
        for fold in 0 1 2; do
          deploy_methods "pancreas_${fold}"
          link="${OUT}/deployment/pancreas_crossfit/fold_${fold}"
          mkdir -p "${link}"
          for linked in splits.parquet graph_smooth safe_fusion; do
            ln -sfn "$(realpath "${source}/${linked}")" "${link}/${linked}"
          done
          for ((i = 1; i < ${#methods[@]}; i += 2)); do
            ln -sfn "$(realpath "${methods[i]#*=}")" "${link}/${methods[i]%%=*}"
          done
        done
        for ((i = 1; i < ${#methods[@]}; i += 2)); do extras+=(--extra-method "${methods[i]%%=*}=${methods[i]%%=*}"); done
        "${PY}" scripts/evaluate_pancreas_crossfit_biology.py --truth "${PAN}/preprocessed.h5ad" \
          --corrupted "${DEPLOY}/pancreas/fold_0/recorded.h5ad" \
          --coordinates "${DEPLOY}/pancreas/fold_0/empty_coordinates.parquet" \
          --crossfit-dir "${OUT}/deployment/pancreas_crossfit" --safe-fusion-subdir safe_fusion "${extras[@]}" \
          --output-dir "${EVAL}/pancreas_biology" --bootstrap 2000 --seed 1729 --marker-panel source
        ;;
      2)
        deploy_methods zebrafish
        "${PY}" scripts/evaluate_trajectory_preservation.py --truth external_data/prepared/zebrafish_trajectory.h5ad \
          --corrupted "${source}/recorded.h5ad" --splits "${source}/splits.parquet" "${methods[@]}" \
          --output-dir "${EVAL}/trajectory/zebrafish" --bootstrap 2000 --seed 1729
        ;;
      3)
        deploy_methods norman_crispra
        "${PY}" scripts/evaluate_interventional_grn.py --dataset norman_crispra --intervention gain_of_function \
          --truth "${NR}/prepared/norman_crispra.h5ad" --corrupted "${source}/recorded.h5ad" \
          --splits "${source}/splits.parquet" "${methods[@]}" --publication-doi 10.1126/science.aax4438 \
          --output-dir "${EVAL}/grn/norman_crispra" --bootstrap 2000 --seed 1729
        ;;
    esac
    ;;
  disease)
    tissues=(pancreas colon)
    "${PY}" scripts/v2_value_disease.py --tissue "${tissues[task]}" --values "${ALTERNATIVES[@]}" \
      --fills-root "${OUT}/deployment/fills" --output-dir "${OUT}/deployment/disease_effects" --jobs "${OMP_NUM_THREADS}"
    ;;
  sle)
    .venv-sle/bin/python scripts/v2_value_sle.py --set main --contract-root "${OUT}/contracts" --values "${VALUES[@]}" \
      --output-dir "${OUT}/sle" --draws 2000
    ;;
  table)
    "${PY}" scripts/v2_value_tables.py --value-root "${OUT}" --values "${ALTERNATIVES[@]}"
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
step "${STAGE}"
