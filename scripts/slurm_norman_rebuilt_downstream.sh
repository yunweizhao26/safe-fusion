#!/usr/bin/env bash
#SBATCH --job-name=sf-norman-rebuilt-down
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-norman-rebuilt-down-%x-%j.out
#SBATCH --error=logs/slurm-norman-rebuilt-down-%x-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

PY=.venv/bin/python
NR="${NORMAN_REBUILT_ROOT:-artifacts/paper_evidence/review_round2/norman_rebuilt}"
LF="${LEAKAGE_FREE_ROOT:-artifacts/paper_evidence/review_round2/leakage_free}"
BENCH="${LF}/norman_crispra"
METHODS="${BENCH}/methods"
SELECTOR="${LF}/selector_mlp_biology_range_fullteachers/norman_crispra"
TRUTH="${NR}/prepared/norman_crispra.h5ad"
MATCHED="${NR}/matched_fraction"
DEPLOY="${NR}/deployment/norman_crispra"
DEPLOY_METHODS="${DEPLOY}/methods"
PCTS=(1 2 3 4 5 6 7 8 9 10)
GRN=(--dataset norman_crispra --intervention gain_of_function --publication-doi 10.1126/science.aax4438 --bootstrap 2000 --seed 1729)

suffix_for_pct() {
  if (( $1 == 10 )); then echo 0p1; else echo "0p0$1"; fi
}

case "${STAGE}" in
  matched)
    for spec in "svd_impute svd" "graph_smooth weighted_knn"; do
      set -- ${spec}
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${BENCH}/corrupted.h5ad" --splits "${BENCH}/splits.parquet" \
        --method-contract "${METHODS}/$1" --method-name "$2" --output-root "${MATCHED}"
    done
    ;;
  masked_grn)
    methods=(
      --method "gene_median=${METHODS}/gene_median"
      --method "svd=${METHODS}/svd_impute"
      --method "weighted_knn=${METHODS}/graph_smooth"
      --method "safe_fusion_dense=${METHODS}/safe_fusion"
    )
    for pct in "${PCTS[@]}"; do
      suffix="$(suffix_for_pct "${pct}")"
      methods+=(--method "safe_fusion_${pct}pct=${SELECTOR}/safe_fusion_calibrated_mlp_topk_${suffix}")
      methods+=(--method "svd_${pct}pct=${MATCHED}/svd_topk_${suffix}")
      methods+=(--method "weighted_knn_${pct}pct=${MATCHED}/weighted_knn_topk_${suffix}")
    done
    "${PY}" scripts/evaluate_interventional_grn.py "${GRN[@]}" \
      --truth "${TRUTH}" --corrupted "${BENCH}/corrupted.h5ad" --splits "${BENCH}/splits.parquet" \
      "${methods[@]}" --output-dir "${NR}/downstream_masked/grn/norman_crispra"
    ;;
  detection_masked|detection_deploy)
    if [[ "${STAGE}" == detection_masked ]]; then
      data=(--corrupted "${BENCH}/corrupted.h5ad" --coordinates "${BENCH}/coordinates.parquet" --splits "${BENCH}/splits.parquet")
      models="${METHODS}"
      output="${NR}/detection_rule/norman_crispra"
    else
      data=(--corrupted "${DEPLOY}/hybrid.h5ad" --coordinates "${DEPLOY}/coordinates.parquet" --splits "${DEPLOY}/splits.parquet")
      models="${DEPLOY_METHODS}"
      output="${DEPLOY}/detection_rule"
    fi
    mapfile -t teacher_args < <(teacher_contract_args "${models}")
    "${PY}" scripts/calibrated_selective_fill.py "${data[@]}" --truth "${TRUTH}" \
      --fusion-contract "${models}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${output}" --fit-split development \
      --architecture mlp --budget-mode apply_topk --budgets 0.05 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
      --detection-rule-mask-rate 0.10 --seed 1729 > /dev/null
    ;;
  deploy_prepare)
    "${PY}" scripts/build_deployment_inputs.py --truth "${TRUTH}" --corrupted "${BENCH}/corrupted.h5ad" \
      --coordinates "${BENCH}/coordinates.parquet" --splits "${BENCH}/splits.parquet" --output-dir "${DEPLOY}"
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${DEPLOY}/hybrid.h5ad" --coordinates "${DEPLOY}/coordinates.parquet" \
        --splits "${DEPLOY}/splits.parquet" --output "${DEPLOY_METHODS}/${method}" --seed 1729
    done
    .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
      --input "${DEPLOY}/hybrid.h5ad" --coordinates "${DEPLOY}/coordinates.parquet" \
      --splits "${DEPLOY}/splits.parquet" --output "${DEPLOY_METHODS}/magic_inductive" \
      --seed 1729 --n-jobs "${SLURM_CPUS_PER_TASK}"
    ;;
  deploy_scvi)
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${DEPLOY}/hybrid.h5ad" --coordinates "${DEPLOY}/coordinates.parquet" \
      --splits "${DEPLOY}/splits.parquet" --output "${DEPLOY_METHODS}/scvi_inductive" --seed 1729
    ;;
  deploy_fill)
    mapfile -t teacher_args < <(teacher_contract_args "${DEPLOY_METHODS}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${DEPLOY}/hybrid.h5ad" --coordinates "${DEPLOY}/coordinates.parquet" \
      --splits "${DEPLOY}/splits.parquet" --output "${DEPLOY_METHODS}/safe_fusion" \
      --seed 1729 "${teacher_args[@]}"
    fractions=()
    for pct in "${PCTS[@]}"; do fractions+=("$(printf '0.%02d' "${pct}")"); done
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${DEPLOY}/hybrid.h5ad" --truth "${TRUTH}" \
      --coordinates "${DEPLOY}/coordinates.parquet" --splits "${DEPLOY}/splits.parquet" \
      --fusion-contract "${DEPLOY_METHODS}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${DEPLOY}/selector" --fit-split development \
      --architecture mlp --budget-mode apply_topk --budgets "${fractions[@]}" \
      --curve-points 2 --seed 1729
    for spec in "svd_impute svd" "graph_smooth weighted_knn"; do
      set -- ${spec}
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${DEPLOY}/hybrid.h5ad" --splits "${DEPLOY}/splits.parquet" \
        --method-contract "${DEPLOY_METHODS}/$1" --method-name "$2" \
        --output-root "${DEPLOY}/matched" --fractions "${fractions[@]}"
    done
    sources=(
      --source "gene_median=${DEPLOY_METHODS}/gene_median"
      --source "svd_impute=${DEPLOY_METHODS}/svd_impute"
      --source "graph_smooth=${DEPLOY_METHODS}/graph_smooth"
      --source "safe_fusion=${DEPLOY_METHODS}/safe_fusion"
    )
    for pct in "${PCTS[@]}"; do
      suffix="$(suffix_for_pct "${pct}")"
      sources+=(--source "safe_fusion_${pct}pct=${DEPLOY}/selector/safe_fusion_calibrated_mlp_topk_${suffix}")
      sources+=(--source "svd_${pct}pct=${DEPLOY}/matched/svd_topk_${suffix}")
      sources+=(--source "weighted_knn_${pct}pct=${DEPLOY}/matched/weighted_knn_topk_${suffix}")
    done
    "${PY}" scripts/finalize_deployment_contracts.py --recorded "${DEPLOY}/recorded.h5ad" \
      --splits "${DEPLOY}/splits.parquet" "${sources[@]}" --output-root "${DEPLOY}"
    ;;
  deploy_eval)
    methods=(
      --method "gene_median=${DEPLOY}/gene_median"
      --method "svd=${DEPLOY}/svd_impute"
      --method "weighted_knn=${DEPLOY}/graph_smooth"
      --method "safe_fusion_dense=${DEPLOY}/safe_fusion"
    )
    for pct in "${PCTS[@]}"; do
      methods+=(--method "safe_fusion_${pct}pct=${DEPLOY}/safe_fusion_${pct}pct")
      methods+=(--method "svd_${pct}pct=${DEPLOY}/svd_${pct}pct")
      methods+=(--method "weighted_knn_${pct}pct=${DEPLOY}/weighted_knn_${pct}pct")
    done
    "${PY}" scripts/evaluate_interventional_grn.py "${GRN[@]}" \
      --truth "${TRUTH}" --corrupted "${DEPLOY}/recorded.h5ad" --splits "${DEPLOY}/splits.parquet" \
      "${methods[@]}" --output-dir "${NR}/deployment/evaluation/grn/norman_crispra"
    "${PY}" scripts/summarize_fill_evaluations.py --root "${NR}/deployment/evaluation" \
      --output "${NR}/deployment/summary.csv"
    ;;
  summarize)
    "${PY}" scripts/summarize_norman_rebuilt_downstream.py --root "${NR}"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
