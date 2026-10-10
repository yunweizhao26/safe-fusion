#!/usr/bin/env bash
#SBATCH --job-name=sf-r3-null
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-r3-null-%x-%A_%a.out
#SBATCH --error=logs/slurm-r3-null-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | prepare | gpu | fill | comparators | correlation}"
design="${2:-permuted}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

PY=.venv/bin/python
R="${NULL_ROOT:-artifacts/paper_evidence/review_round3/downstream/correlation_null}"

R3_DEPLOY="${R3_DEPLOY:-artifacts/paper_evidence/review_round3/downstream/deployment}"
KEYS=(colon pancreas_0 pancreas_1 pancreas_2)
FRACTIONS=(0.01 0.05 0.10)
PCTS=(1 5 10)
step() { echo "step=$1 elapsed_s=${SECONDS}"; }
suffix_for_pct() {
  if (( $1 == 10 )); then echo 0p1; else echo "0p0$1"; fi
}

unit_relative() {
  case "$1" in
    pancreas_*) echo "pancreas/fold_${1##*_}" ;;
    *) echo "$1" ;;
  esac
}

paths_for() {
  deployment_paths "$1"
  perm="${R}/permuted/$(unit_relative "$1")"
  perm_methods="${perm}/methods"
}

case "${stage}" in
  submit)
    mkdir -p logs
    prep=$(sbatch --parsable --job-name=sf-r3null-prep --array=0-3 -p cs,cpu_short "${SCRIPT}" prepare)
    gpu=$(sbatch --parsable --job-name=sf-r3null-gpu --array=0-3 --mem=32G --gres=gpu:l40s:1 --dependency=afterok:${prep} "${SCRIPT}" gpu)
    fill=$(sbatch --parsable --job-name=sf-r3null-fill --array=0-3 -p cs --dependency=afterok:${gpu} "${SCRIPT}" fill)
    corr=$(sbatch --parsable --job-name=sf-r3null-corr -p cs,cpu_short --mem=64G --time=03:00:00 --dependency=afterok:${fill} "${SCRIPT}" correlation permuted)
    echo "prepare=${prep} gpu=${gpu} fill=${fill} correlation=${corr}"
    ;;
  prepare)
    key="${KEYS[SLURM_ARRAY_TASK_ID]}"
    paths_for "${key}"
    "${PY}" scripts/r3_downstream_permute.py --deployment-dir "${out}" --label-column cell_type \
      --seed 1729 --output-dir "${perm}"
    step permute
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${perm}/hybrid.h5ad" --coordinates "${perm}/coordinates.parquet" \
        --splits "${perm}/splits.parquet" --output "${perm_methods}/${method}" --seed 1729
      step "${method}"
    done
    .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
      --input "${perm}/hybrid.h5ad" --coordinates "${perm}/coordinates.parquet" \
      --splits "${perm}/splits.parquet" --output "${perm_methods}/magic_inductive" \
      --seed 1729 --n-jobs "${SLURM_CPUS_PER_TASK}"
    step magic_inductive
    .conda-magic-current/bin/python scripts/run_magic_baseline.py \
      --corrupted "${perm}/hybrid.h5ad" --coordinates "${perm}/coordinates.parquet" \
      --splits "${perm}/splits.parquet" --output "${perm_methods}/magic" \
      --n-jobs "${SLURM_CPUS_PER_TASK}" --seed 1729
    step magic_standard
    ;;
  gpu)
    key="${KEYS[SLURM_ARRAY_TASK_ID]}"
    paths_for "${key}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${perm}/hybrid.h5ad" --coordinates "${perm}/coordinates.parquet" \
      --splits "${perm}/splits.parquet" --output "${perm_methods}/scvi_inductive" --seed 1729
    step scvi_inductive
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${perm}/hybrid.h5ad" --coordinates "${perm}/coordinates.parquet" \
      --splits "${perm}/splits.parquet" --output "${perm_methods}/scvi" --epochs 200 --seed 1729
    step scvi_standard
    ;;
  fill)
    key="${KEYS[SLURM_ARRAY_TASK_ID]}"
    paths_for "${key}"
    mapfile -t teacher_args < <(teacher_contract_args "${perm_methods}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${perm}/hybrid.h5ad" --coordinates "${perm}/coordinates.parquet" \
      --splits "${perm}/splits.parquet" --output "${perm_methods}/safe_fusion" \
      --seed 1729 "${teacher_args[@]}"
    step safe_fusion
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${perm}/hybrid.h5ad" --truth "${truth}" \
      --coordinates "${perm}/coordinates.parquet" --splits "${perm}/splits.parquet" \
      --fusion-contract "${perm_methods}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${perm}/selector" "${fit[@]}" \
      --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
      --curve-points 2 --seed 1729
    step selector
    for spec in "svd_impute svd" "graph_smooth weighted_knn"; do
      set -- ${spec}
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${perm}/hybrid.h5ad" --splits "${perm}/splits.parquet" \
        --method-contract "${perm_methods}/$1" --method-name "$2" \
        --output-root "${perm}/matched" --fractions "${FRACTIONS[@]}"
    done
    for name in magic scvi; do
      "${PY}" scripts/count_scale_contract.py --contract "${perm_methods}/${name}" \
        --corrupted "${perm}/hybrid.h5ad" --output "${perm_methods}/${name}_counts"
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${perm}/hybrid.h5ad" --splits "${perm}/splits.parquet" \
        --method-contract "${perm_methods}/${name}_counts" --method-name "${name}" \
        --output-root "${perm}/matched" --fractions "${FRACTIONS[@]}"
    done
    step matched
    sources=()
    for pct in "${PCTS[@]}"; do
      suffix="$(suffix_for_pct "${pct}")"
      sources+=(--source "safe_fusion_${pct}pct=${perm}/selector/safe_fusion_calibrated_mlp_topk_${suffix}")
      for name in svd weighted_knn magic scvi; do
        sources+=(--source "${name}_${pct}pct=${perm}/matched/${name}_topk_${suffix}")
      done
    done
    "${PY}" scripts/finalize_deployment_contracts.py --recorded "${perm}/recorded.h5ad" \
      --splits "${perm}/splits.parquet" "${sources[@]}" --output-root "${perm}"
    step finalize
    ;;
  comparators)

    key="${KEYS[SLURM_ARRAY_TASK_ID]}"
    paths_for "${key}"
    .venv-scanpy/bin/python scripts/standard_imputers/r3_kcluster.py --corrupted "${perm}/hybrid.h5ad" \
      --output "${perm_methods}/kcluster.json" --seed 1729
    for name in dca scimpute; do
      .venv/bin/python -u scripts/standard_imputers/run_r3_comparator.py --method "${name}" \
        --corrupted "${perm}/hybrid.h5ad" --coordinates "${perm}/coordinates.parquet" --splits "${perm}/splits.parquet" \
        --output "${perm_methods}/${name}" --kcluster-json "${perm_methods}/kcluster.json" \
        --ncores "${SLURM_CPUS_PER_TASK}" --seed 1729
      "${PY}" scripts/count_scale_contract.py --contract "${perm_methods}/${name}" \
        --corrupted "${perm}/hybrid.h5ad" --output "${perm_methods}/${name}_counts"
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${perm}/hybrid.h5ad" --splits "${perm}/splits.parquet" \
        --method-contract "${perm_methods}/${name}_counts" --method-name "${name}" \
        --output-root "${perm}/matched" --fractions "${FRACTIONS[@]}"
      sources=()
      for pct in "${PCTS[@]}"; do
        sources+=(--source "${name}_${pct}pct=${perm}/matched/${name}_topk_$(suffix_for_pct "${pct}")")
      done
      "${PY}" scripts/finalize_deployment_contracts.py --recorded "${perm}/recorded.h5ad" \
        --splits "${perm}/splits.parquet" "${sources[@]}" --output-root "${perm}/comparator_counts/${name}"
      for pct in "${PCTS[@]}"; do mv "${perm}/comparator_counts/${name}/${name}_${pct}pct" "${perm}/${name}_${pct}pct"; done
      step "${name}"
    done
    ;;
  correlation)
    [[ "${design}" =~ ^(permuted|recorded)(_comparators)?$ ]] || { echo "unknown design ${design}" >&2; exit 2; }
    units=()
    methods=()
    for key in "${KEYS[@]}"; do
      rel="$(unit_relative "${key}")"
      if [[ "${design}" == permuted* ]]; then
        units+=(--unit "${rel}=${R}/permuted/${rel}")
      else
        units+=(--unit "${rel}=${DEPLOY_ROOT}/${rel}")
      fi
    done
    for pct in "${PCTS[@]}"; do
      for name in safe_fusion svd weighted_knn; do
        methods+=(--method "${name}_${pct}pct={unit}/${name}_${pct}pct")
      done
      if [[ "${design}" == *_comparators ]]; then

        for name in dca scimpute enimpute; do
          present=1
          for key in "${KEYS[@]}"; do
            rel="$(unit_relative "${key}")"
            if [[ "${design}" == permuted* ]]; then dir="${R}/permuted/${rel}"; else dir="${R3_DEPLOY}/${rel}"; fi
            [[ -d "${dir}/${name}_${pct}pct" ]] || present=0
          done
          (( present )) || continue
          if [[ "${design}" == permuted* ]]; then
            methods+=(--method "${name}_${pct}pct={unit}/${name}_${pct}pct")
          else
            methods+=(--method "${name}_${pct}pct=${R3_DEPLOY}/{key}/${name}_${pct}pct")
          fi
        done
        continue
      fi
      for name in magic scvi; do
        if [[ "${design}" == permuted* ]]; then
          methods+=(--method "${name}_${pct}pct={unit}/${name}_${pct}pct")
        else
          methods+=(--method "${name}_${pct}pct=${R3_DEPLOY}/{key}/${name}_${pct}pct")
        fi
      done
    done

    for scale in counts log_cp10k; do
      "${PY}" scripts/r3_downstream_correlation.py "${units[@]}" "${methods[@]}" \
        --label-column cell_type --scale "${scale}" --fdr 0.05 --bootstrap 2000 --seed 1729 \
        --output-dir "${R}/evaluation/${design}/${scale}"
      step "correlation_${design}_${scale}"
    done
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
