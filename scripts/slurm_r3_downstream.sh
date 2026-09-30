#!/usr/bin/env bash
#SBATCH --job-name=sf-r3-down
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-r3-downstream-%x-%A_%a.out
#SBATCH --error=logs/slurm-r3-downstream-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | magic | scvi | fill | eval_deploy | eval_masked | decompose | disease | annotation | comparator_fill | eval_comparators | eval_masked_comparators | summarize}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

PY=.venv/bin/python
R3="${R3_DOWNSTREAM_ROOT:-artifacts/paper_evidence/review_round3/downstream}"
COMPARATORS="${R3_COMPARATORS_ROOT:-artifacts/paper_evidence/review_round3/comparators}"
DEPLOY=artifacts/paper_evidence/downstream_deployment
NR=artifacts/paper_evidence/review_round2/norman_rebuilt
LF=artifacts/paper_evidence/review_round2/leakage_free
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial
COL_METHODS=artifacts/colon_runs/0b2469810675-c0db6f963e94/methods/standardized/colon_epithelial/mask_010
CF=artifacts/paper_evidence/pancreas_crossfit
ZF=artifacts/external_trajectory/zebrafish
BASE=artifacts/paper_evidence/baselines
NORMAN_TRUTH="${NR}/prepared/norman_crispra.h5ad"
UNITS=(colon pancreas_0 pancreas_1 pancreas_2 zebrafish norman_crispra)
STANDARD=(magic scvi)
PCTS=(1 5 10)
FRACTIONS=(0.01 0.05 0.10)
GRN=(--dataset norman_crispra --intervention gain_of_function --publication-doi 10.1126/science.aax4438 --bootstrap 2000 --seed 1729)

suffix_for_pct() {
  if (( $1 == 10 )); then echo 0p1; else echo "0p0$1"; fi
}

deploy_unit() {
  case "$1" in
    pancreas_*) source="${DEPLOY}/pancreas/fold_${1##*_}"; out="${R3}/deployment/pancreas/fold_${1##*_}" ;;
    norman_crispra) source="${NR}/deployment/norman_crispra"; out="${R3}/deployment/norman_crispra" ;;
    *) source="${DEPLOY}/$1"; out="${R3}/deployment/$1" ;;
  esac
}

masked_unit() {
  case "$1" in
    colon)
      corrupted="${COL}/corrupted/mask_010.h5ad"; coordinates="${COL}/coordinates/mask_010.parquet"
      splits="${COL}/splits.parquet"; fits_root="${BASE}/%s/colon"
      selector=artifacts/paper_evidence/selector_mlp_biology_range/colon
      matched=artifacts/paper_evidence/matched_fraction/colon; out="${R3}/masked/colon"
      ;;
    pancreas_*)
      local fold="${1##*_}"
      corrupted="${PAN}/corrupted/mask_010.h5ad"; coordinates="${PAN}/coordinates/mask_010.parquet"
      splits="${CF}/fold_${fold}/splits.parquet"; fits_root="${BASE}/%s/pancreas"
      selector="${CF}/fold_${fold}/selector_mlp_biology_range_fullteachers"
      matched="${CF}/fold_${fold}/matched_fraction"; out="${R3}/masked/pancreas/fold_${fold}"
      ;;
    zebrafish)
      corrupted="${ZF}/corrupted.h5ad"; coordinates="${ZF}/coordinates.parquet"
      splits="${ZF}/splits.parquet"; fits_root="${R3}/masked/fits/zebrafish/%s"
      selector="${ZF}/selector_mlp_biology_range"; matched="${ZF}/matched_fraction"; out="${R3}/masked/zebrafish"
      ;;
    norman_crispra)
      corrupted="${LF}/norman_crispra/corrupted.h5ad"; coordinates="${LF}/norman_crispra/coordinates.parquet"
      splits="${LF}/norman_crispra/splits.parquet"; fits_root="${LF}/baselines/%s/norman"
      selector="${LF}/selector_mlp_biology_range_fullteachers/norman_crispra"
      matched="${NR}/matched_fraction"; out="${R3}/masked/norman_crispra"
      ;;
  esac
}

fit_target() {
  local id="$1"
  if (( id < 6 )); then
    deploy_unit "${UNITS[id]}"
    input="${source}/hybrid.h5ad"; coordinates="${source}/coordinates.parquet"; splits="${source}/splits.parquet"
    fits="${out}/fits"
  else
    masked_unit zebrafish
    input="${corrupted}"; fits="${R3}/masked/fits/zebrafish"
  fi
}

COMPARATOR_METHODS=(dca scimpute enimpute)

deploy_methods() {
  local source="$1" out="$2" set="$3" pct name
  methods=()
  for pct in "${PCTS[@]}"; do
    if [[ "${set}" == comparators ]]; then
      for name in "${COMPARATOR_METHODS[@]}"; do
        [[ -d "${out}/${name}_${pct}pct" ]] && methods+=(--method "${name}_${pct}pct=${out}/${name}_${pct}pct")
      done
      continue
    fi
    methods+=(--method "safe_fusion_${pct}pct=${source}/safe_fusion_${pct}pct")
    methods+=(--method "svd_${pct}pct=${source}/svd_${pct}pct")
    methods+=(--method "weighted_knn_${pct}pct=${source}/weighted_knn_${pct}pct")
    for name in "${STANDARD[@]}"; do methods+=(--method "${name}_${pct}pct=${out}/${name}_${pct}pct"); done
  done
}

masked_methods() {
  local pct suffix name
  methods=()
  if [[ "${1:-}" == comparators ]]; then
    for pct in "${PCTS[@]}"; do
      suffix="$(suffix_for_pct "${pct}")"
      for name in "${COMPARATOR_METHODS[@]}"; do
        [[ -d "${out}/matched/${name}_topk_${suffix}" ]] && methods+=(--method "${name}_${pct}pct=${out}/matched/${name}_topk_${suffix}")
      done
    done
    return 0
  fi
  for pct in "${PCTS[@]}"; do
    suffix="$(suffix_for_pct "${pct}")"
    methods+=(--method "safe_fusion_${pct}pct=${selector}/safe_fusion_calibrated_mlp_topk_${suffix}")
    methods+=(--method "svd_${pct}pct=${matched}/svd_topk_${suffix}")
    methods+=(--method "weighted_knn_${pct}pct=${matched}/weighted_knn_topk_${suffix}")
    for name in "${STANDARD[@]}"; do methods+=(--method "${name}_${pct}pct=${out}/matched/${name}_topk_${suffix}"); done
  done
  for name in "${STANDARD[@]}"; do methods+=(--method "${name}_dense=${out}/${name}_counts"); done
}

link_crossfit() {
  local root="$1" kind="$2" fold name path
  for fold in 0 1 2; do
    local dir="${root}/fold_${fold}"
    mkdir -p "${dir}"
    if [[ "${kind}" != masked ]]; then
      deploy_unit "pancreas_${fold}"
      local base="${source}"
      ln -sfn "$(realpath "${base}/splits.parquet")" "${dir}/splits.parquet"
      ln -sfn "$(realpath "${base}/graph_smooth")" "${dir}/graph_smooth"
      ln -sfn "$(realpath "${base}/safe_fusion")" "${dir}/safe_fusion"
      deploy_methods "${base}" "${out}" "${kind}"
    else
      masked_unit "pancreas_${fold}"
      ln -sfn "$(realpath "${splits}")" "${dir}/splits.parquet"
      ln -sfn "$(realpath "${CF}/fold_${fold}/graph_smooth")" "${dir}/graph_smooth"
      ln -sfn "$(realpath "${CF}/fold_${fold}/safe_fusion")" "${dir}/safe_fusion"
      masked_methods
    fi
    for ((i = 1; i < ${#methods[@]}; i += 2)); do
      name="${methods[i]%%=*}"; path="${methods[i]#*=}"
      [[ "${dir}/${name}" -ef "${path}" ]] || ln -sfn "$(realpath "${path}")" "${dir}/${name}"
    done
  done
  extras=()
  for ((i = 1; i < ${#methods[@]}; i += 2)); do
    name="${methods[i]%%=*}"
    extras+=(--extra-method "${name}=${name}")
  done
}

case "${stage}" in
  submit)
    mkdir -p logs
    magic=$(sbatch --parsable --job-name=sf-r3-magic --array=0-6 --mem=64G "${SCRIPT}" magic)
    scvi=$(sbatch --parsable --job-name=sf-r3-scvi --array=0-6 --gres=gpu:l40s:1 --mem=32G "${SCRIPT}" scvi)
    fill=$(sbatch --parsable --job-name=sf-r3-fill --array=0-11 --dependency=afterok:${magic}:${scvi} "${SCRIPT}" fill)
    deval=$(sbatch --parsable --job-name=sf-r3-deval --array=0-5 --time=06:00:00 --dependency=afterok:${fill} "${SCRIPT}" eval_deploy)
    meval=$(sbatch --parsable --job-name=sf-r3-meval --array=0-5 --time=06:00:00 --dependency=afterok:${fill} "${SCRIPT}" eval_masked)
    decomp=$(sbatch --parsable --job-name=sf-r3-decomp --array=0-5 --time=04:00:00 --dependency=afterok:${fill} "${SCRIPT}" decompose)
    disease=$(sbatch --parsable --job-name=sf-r3-disease --array=0-1 --time=03:00:00 --dependency=afterok:${fill} "${SCRIPT}" disease)
    annot=$(sbatch --parsable --job-name=sf-r3-annot --array=0-1 --time=03:00:00 --dependency=afterok:${fill} "${SCRIPT}" annotation)
    summ=$(sbatch --parsable --job-name=sf-r3-summ --cpus-per-task=2 --mem=16G --time=00:30:00 \
      --dependency=afterok:${deval}:${meval}:${decomp}:${disease}:${annot} "${SCRIPT}" summarize)
    echo "magic=${magic} scvi=${scvi} fill=${fill} eval_deploy=${deval} eval_masked=${meval} decompose=${decomp} disease=${disease} annotation=${annot} summarize=${summ}"
    ;;
  magic)
    fit_target "${SLURM_ARRAY_TASK_ID}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${fits}/magic" \
      --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    echo "elapsed_s=${SECONDS}"
    ;;
  scvi)
    fit_target "${SLURM_ARRAY_TASK_ID}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${fits}/scvi" --epochs 200 --seed 1729
    echo "elapsed_s=${SECONDS}"
    ;;
  fill)

    id="${SLURM_ARRAY_TASK_ID}"
    if (( id < 6 )); then
      deploy_unit "${UNITS[id]}"
      sources=()
      for name in "${STANDARD[@]}"; do
        "${PY}" scripts/count_scale_contract.py --contract "${out}/fits/${name}" \
          --corrupted "${source}/hybrid.h5ad" --output "${out}/fits/${name}_counts"
        "${PY}" scripts/apply_fill_fraction.py --corrupted "${source}/hybrid.h5ad" --splits "${source}/splits.parquet" \
          --method-contract "${out}/fits/${name}_counts" --method-name "${name}" \
          --output-root "${out}/matched" --fractions "${FRACTIONS[@]}"
        for pct in "${PCTS[@]}"; do
          sources+=(--source "${name}_${pct}pct=${out}/matched/${name}_topk_$(suffix_for_pct "${pct}")")
        done
      done
      "${PY}" scripts/finalize_deployment_contracts.py --recorded "${source}/recorded.h5ad" \
        --splits "${source}/splits.parquet" "${sources[@]}" --output-root "${out}"
    else
      masked_unit "${UNITS[id - 6]}"
      for name in "${STANDARD[@]}"; do
        "${PY}" scripts/count_scale_contract.py --contract "$(printf "${fits_root}" "${name}")" \
          --corrupted "${corrupted}" --output "${out}/${name}_counts"
        "${PY}" scripts/apply_fill_fraction.py --corrupted "${corrupted}" --splits "${splits}" \
          --method-contract "${out}/${name}_counts" --method-name "${name}" \
          --output-root "${out}/matched" --fractions "${FRACTIONS[@]}"
      done
    fi
    echo "elapsed_s=${SECONDS}"
    ;;
  eval_deploy|eval_masked|eval_comparators|eval_masked_comparators)
    id="${SLURM_ARRAY_TASK_ID}"
    kind="${stage#eval_}"
    set_arg=""
    case "${kind}" in
      deploy) evaluation="${R3}/deployment/evaluation" ;;
      comparators) evaluation="${R3}/deployment/evaluation_comparators" ;;
      masked) evaluation="${R3}/masked/evaluation" ;;
      masked_comparators) evaluation="${R3}/masked/evaluation_comparators"; kind=masked; set_arg=comparators ;;
    esac
    case "${id}" in
      0|2)
        if [[ "${kind}" != masked ]]; then
          deploy_unit colon; deploy_methods "${source}" "${out}" "${kind}"
          truth="${COL}/preprocessed.h5ad"; corrupted="${source}/recorded.h5ad"
          coordinates="${source}/empty_coordinates.parquet"; splits="${source}/splits.parquet"
        else
          masked_unit colon; masked_methods ${set_arg}; truth="${COL}/preprocessed.h5ad"
        fi
        if (( id == 0 )); then
          "${PY}" scripts/evaluate_unsupervised_clustering.py --truth "${truth}" --corrupted "${corrupted}" \
            --splits "${splits}" "${methods[@]}" --label-column cell_type --unit-column donor \
            --output-dir "${evaluation}/clustering/colon" --cluster-seeds 20 --bootstrap 2000 --seed 1729 \
            --allow-transductive
        else
          "${PY}" scripts/evaluate_colon_donor_biology.py --truth "${truth}" --corrupted "${corrupted}" \
            --coordinates "${coordinates}" --splits "${splits}" "${methods[@]}" \
            --output-dir "${evaluation}/markers/colon" --bootstrap 2000 --seed 1729 --marker-panel source \
            --allow-transductive
        fi
        ;;
      1)
        fold_outputs=()
        for fold in 0 1 2; do
          if [[ "${kind}" != masked ]]; then
            deploy_unit "pancreas_${fold}"; deploy_methods "${source}" "${out}" "${kind}"
            corrupted="${source}/recorded.h5ad"; splits="${source}/splits.parquet"
          else
            masked_unit "pancreas_${fold}"; masked_methods
          fi
          fold_outputs+=(--fold "${evaluation}/clustering/pancreas/fold_${fold}")
          "${PY}" scripts/evaluate_unsupervised_clustering.py --truth "${PAN}/preprocessed.h5ad" \
            --corrupted "${corrupted}" --splits "${splits}" "${methods[@]}" --label-column cell_type \
            --unit-column donor --output-dir "${evaluation}/clustering/pancreas/fold_${fold}" \
            --cluster-seeds 20 --bootstrap 500 --seed "$((1729 + fold))" --allow-transductive
        done
        "${PY}" scripts/combine_clustering_folds.py "${fold_outputs[@]}" \
          --output-dir "${evaluation}/clustering/pancreas/combined" --bootstrap 2000 --seed 1729
        ;;
      3)
        if [[ "${kind}" != masked ]]; then
          crossfit="${R3}/deployment/pancreas_crossfit"
          [[ "${kind}" == comparators ]] && crossfit="${R3}/deployment/pancreas_crossfit_comparators"
          link_crossfit "${crossfit}" "${kind}"
          deploy_unit pancreas_0
          data=(--corrupted "${source}/recorded.h5ad" --coordinates "${source}/empty_coordinates.parquet"
                --crossfit-dir "${crossfit}")
        else
          link_crossfit "${R3}/masked/pancreas_crossfit" masked
          data=(--corrupted "${PAN}/corrupted/mask_010.h5ad" --coordinates "${PAN}/coordinates/mask_010.parquet"
                --crossfit-dir "${R3}/masked/pancreas_crossfit")
        fi
        "${PY}" scripts/evaluate_pancreas_crossfit_biology.py --truth "${PAN}/preprocessed.h5ad" "${data[@]}" \
          --safe-fusion-subdir safe_fusion "${extras[@]}" --output-dir "${evaluation}/pancreas_biology" \
          --bootstrap 2000 --seed 1729 --marker-panel source --allow-transductive
        ;;
      4)
        if [[ "${kind}" != masked ]]; then
          deploy_unit zebrafish; deploy_methods "${source}" "${out}" "${kind}"
          corrupted="${source}/recorded.h5ad"; splits="${source}/splits.parquet"
        else
          masked_unit zebrafish; masked_methods
        fi
        "${PY}" scripts/evaluate_trajectory_preservation.py --truth external_data/prepared/zebrafish_trajectory.h5ad \
          --corrupted "${corrupted}" --splits "${splits}" "${methods[@]}" \
          --output-dir "${evaluation}/trajectory/zebrafish" --bootstrap 2000 --seed 1729 --allow-transductive
        ;;
      5)
        if [[ "${kind}" != masked ]]; then
          deploy_unit norman_crispra; deploy_methods "${source}" "${out}" "${kind}"
          corrupted="${source}/recorded.h5ad"; splits="${source}/splits.parquet"
        else
          masked_unit norman_crispra; masked_methods ${set_arg}
        fi
        "${PY}" scripts/evaluate_interventional_grn.py "${GRN[@]}" --truth "${NORMAN_TRUTH}" \
          --corrupted "${corrupted}" --splits "${splits}" "${methods[@]}" \
          --output-dir "${evaluation}/grn/norman_crispra" --allow-transductive
        ;;
    esac
    echo "elapsed_s=${SECONDS}"
    ;;
  comparator_fill)

    if (( SLURM_ARRAY_TASK_ID >= 6 )); then
      masked_units=(colon norman_crispra)
      key="${masked_units[SLURM_ARRAY_TASK_ID - 6]}"
      masked_unit "${key}"
      for name in "${COMPARATOR_METHODS[@]}"; do
        fit="${COMPARATORS}/fits/${name}/${key}"
        [[ -f "${fit}/mean.npy" && ! -d "${out}/matched/${name}_topk_0p1" ]] || continue
        "${PY}" scripts/count_scale_contract.py --contract "${fit}" --corrupted "${corrupted}" --output "${out}/${name}_counts"
        "${PY}" scripts/apply_fill_fraction.py --corrupted "${corrupted}" --splits "${splits}" \
          --method-contract "${out}/${name}_counts" --method-name "${name}" \
          --output-root "${out}/matched" --fractions "${FRACTIONS[@]}"
      done
      echo "elapsed_s=${SECONDS}"
      exit 0
    fi
    deploy_unit "${UNITS[SLURM_ARRAY_TASK_ID]}"
    for name in "${COMPARATOR_METHODS[@]}"; do
      fit="${COMPARATORS}/fits/${name}/deploy_${UNITS[SLURM_ARRAY_TASK_ID]}"
      [[ -f "${fit}/mean.npy" && ! -d "${out}/${name}_10pct" ]] || continue
      "${PY}" scripts/count_scale_contract.py --contract "${fit}" --corrupted "${source}/hybrid.h5ad" \
        --output "${out}/fits/${name}_counts"
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${source}/hybrid.h5ad" --splits "${source}/splits.parquet" \
        --method-contract "${out}/fits/${name}_counts" --method-name "${name}" \
        --output-root "${out}/matched" --fractions "${FRACTIONS[@]}"
      sources=()
      for pct in "${PCTS[@]}"; do
        sources+=(--source "${name}_${pct}pct=${out}/matched/${name}_topk_$(suffix_for_pct "${pct}")")
      done
      "${PY}" scripts/finalize_deployment_contracts.py --recorded "${source}/recorded.h5ad" \
        --splits "${source}/splits.parquet" "${sources[@]}" --output-root "${out}/comparator_counts/${name}"
      for pct in "${PCTS[@]}"; do mv "${out}/comparator_counts/${name}/${name}_${pct}pct" "${out}/${name}_${pct}pct"; done
    done
    echo "elapsed_s=${SECONDS}"
    ;;
  decompose)

    datasets=(colon pancreas zebrafish)
    dataset="${datasets[SLURM_ARRAY_TASK_ID / 2]}"
    name="${STANDARD[SLURM_ARRAY_TASK_ID % 2]}"
    root="${R3}/masked/decomposition/${dataset}/${name}"
    decompose() {
      local sources_args=() pct suffix fill
      methods=()
      for pct in "${PCTS[@]}"; do
        suffix="$(suffix_for_pct "${pct}")"; fill="${out}/matched/${name}_topk_${suffix}"
        sources_args+=(--source "${name}_${pct}pct=${fill}")
        methods+=(--method "${name}_${pct}pct=${fill}")
        methods+=(--method "${name}_${pct}pct__masked_only=$1/${name}_${pct}pct__masked_only")
        methods+=(--method "${name}_${pct}pct__zeros_only=$1/${name}_${pct}pct__zeros_only")
      done
      "${PY}" scripts/decompose_fills.py --corrupted "${corrupted}" --coordinates "${coordinates}" \
        --splits "${splits}" "${sources_args[@]}" --output-root "$1"
    }
    case "${dataset}" in
      colon)
        masked_unit colon; decompose "${root}"
        "${PY}" scripts/evaluate_colon_donor_biology.py --truth "${COL}/preprocessed.h5ad" --corrupted "${corrupted}" \
          --coordinates "${coordinates}" --splits "${splits}" "${methods[@]}" \
          --output-dir "${root}/evaluation/markers" --bootstrap 2000 --seed 1729 --marker-panel source --allow-transductive
        "${PY}" scripts/evaluate_unsupervised_clustering.py --truth "${COL}/preprocessed.h5ad" --corrupted "${corrupted}" \
          --splits "${splits}" "${methods[@]}" --label-column cell_type --unit-column donor \
          --output-dir "${root}/evaluation/clustering" --cluster-seeds 20 --bootstrap 2000 --seed 1729 --allow-transductive
        ;;
      pancreas)
        fold_outputs=(); extras=()
        for fold in 0 1 2; do
          masked_unit "pancreas_${fold}"
          fold_root="${root}/fold_${fold}"
          decompose "${fold_root}"
          for linked in graph_smooth safe_fusion; do
            ln -sfn "$(realpath "${CF}/fold_${fold}/${linked}")" "${fold_root}/${linked}"
          done
          ln -sfn "$(realpath "${splits}")" "${fold_root}/splits.parquet"
          for pct in "${PCTS[@]}"; do
            ln -sfn "$(realpath "${out}/matched/${name}_topk_$(suffix_for_pct "${pct}")")" "${fold_root}/${name}_${pct}pct"
          done
          fold_outputs+=(--fold "${root}/evaluation/clustering/fold_${fold}")
          "${PY}" scripts/evaluate_unsupervised_clustering.py --truth "${PAN}/preprocessed.h5ad" --corrupted "${corrupted}" \
            --splits "${splits}" "${methods[@]}" --label-column cell_type --unit-column donor \
            --output-dir "${root}/evaluation/clustering/fold_${fold}" --cluster-seeds 20 --bootstrap 500 \
            --seed "$((1729 + fold))" --allow-transductive
        done
        "${PY}" scripts/combine_clustering_folds.py "${fold_outputs[@]}" \
          --output-dir "${root}/evaluation/clustering/combined" --bootstrap 2000 --seed 1729
        for pct in "${PCTS[@]}"; do
          for part in "" __masked_only __zeros_only; do
            extras+=(--extra-method "${name}_${pct}pct${part}=${name}_${pct}pct${part}")
          done
        done
        "${PY}" scripts/evaluate_pancreas_crossfit_biology.py --truth "${PAN}/preprocessed.h5ad" \
          --corrupted "${corrupted}" --coordinates "${coordinates}" --crossfit-dir "${root}" \
          --safe-fusion-subdir safe_fusion "${extras[@]}" --output-dir "${root}/evaluation/biology" \
          --bootstrap 2000 --seed 1729 --marker-panel source --allow-transductive
        ;;
      zebrafish)
        masked_unit zebrafish; decompose "${root}"
        "${PY}" scripts/evaluate_trajectory_preservation.py --truth external_data/prepared/zebrafish_trajectory.h5ad \
          --corrupted "${corrupted}" --splits "${splits}" "${methods[@]}" \
          --output-dir "${root}/evaluation/trajectory" --bootstrap 2000 --seed 1729 --allow-transductive
        ;;
    esac
    echo "elapsed_s=${SECONDS}"
    ;;
  disease)
    tissues=(pancreas colon)
    "${PY}" scripts/r3_downstream_disease.py --tissue "${tissues[SLURM_ARRAY_TASK_ID]}" \
      --root "${R3}" --jobs "${OMP_NUM_THREADS}" ${COMPARATORS_ONLY:+--comparators-only}
    echo "elapsed_s=${SECONDS}"
    ;;
  annotation)
    tissues=(pancreas colon)
    .venv-scanpy/bin/python scripts/r3_downstream_annotation.py --tissue "${tissues[SLURM_ARRAY_TASK_ID]}" \
      --root "${R3}" ${COMPARATORS_ONLY:+--comparators-only}
    echo "elapsed_s=${SECONDS}"
    ;;
  summarize)
    "${PY}" scripts/r3_downstream_tables.py --root "${R3}"
    echo "elapsed_s=${SECONDS}"
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
