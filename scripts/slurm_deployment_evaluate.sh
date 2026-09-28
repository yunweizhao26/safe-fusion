#!/usr/bin/env bash
#SBATCH --job-name=sf-deploy-eval
#SBATCH --account=torch_pr_634_general
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-8%4
#SBATCH --output=logs/slurm-deployment-evaluate-%A_%a.out
#SBATCH --error=logs/slurm-deployment-evaluate-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
ROOT="${DEPLOY_ROOT:-artifacts/paper_evidence/downstream_deployment}"
OUT="${ROOT}/evaluation"
PCTS="${PCTS:-1 2 3 4 5 6 7 8 9 10}"
MARKER_PANEL="${MARKER_PANEL:-source}"

methods_for() {
  local unit="$1" pct
  methods=(
    --method "gene_median=${unit}/gene_median"
    --method "svd=${unit}/svd_impute"
    --method "weighted_knn=${unit}/graph_smooth"
    --method "safe_fusion_dense=${unit}/safe_fusion"
  )
  for pct in ${PCTS}; do
    methods+=(--method "safe_fusion_${pct}pct=${unit}/safe_fusion_${pct}pct")
    methods+=(--method "svd_${pct}pct=${unit}/svd_${pct}pct")
    methods+=(--method "weighted_knn_${pct}pct=${unit}/weighted_knn_${pct}pct")
  done
}

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    unit="${ROOT}/colon"
    methods_for "${unit}"
    "${PY}" scripts/evaluate_unsupervised_clustering.py \
      --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" --corrupted "${unit}/recorded.h5ad" \
      --splits "${unit}/splits.parquet" "${methods[@]}" --label-column cell_type --unit-column donor \
      --output-dir "${OUT}/clustering/colon" --cluster-seeds 20 --bootstrap 2000 --seed 1729
    ;;
  1)
    fold_outputs=()
    for fold in 0 1 2; do
      unit="${ROOT}/pancreas/fold_${fold}"
      methods_for "${unit}"
      fold_outputs+=(--fold "${OUT}/clustering/pancreas/fold_${fold}")
      "${PY}" scripts/evaluate_unsupervised_clustering.py \
        --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" --corrupted "${unit}/recorded.h5ad" \
        --splits "${unit}/splits.parquet" "${methods[@]}" --label-column cell_type --unit-column donor \
        --output-dir "${OUT}/clustering/pancreas/fold_${fold}" --cluster-seeds 20 --bootstrap 500 \
        --seed "$((1729 + fold))"
    done
    "${PY}" scripts/combine_clustering_folds.py "${fold_outputs[@]}" \
      --output-dir "${OUT}/clustering/pancreas/combined" --bootstrap 2000 --seed 1729
    ;;
  2)
    unit="${ROOT}/colon"
    methods_for "${unit}"
    "${PY}" scripts/evaluate_colon_donor_biology.py \
      --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" --corrupted "${unit}/recorded.h5ad" \
      --coordinates "${unit}/empty_coordinates.parquet" --splits "${unit}/splits.parquet" \
      "${methods[@]}" --output-dir "${OUT}/markers/colon" --bootstrap 2000 --seed 1729 \
      --marker-panel "${MARKER_PANEL}"
    ;;
  3)
    extras=(--extra-method "gene_median=gene_median" --extra-method "svd=svd_impute")
    for pct in ${PCTS}; do
      extras+=(--extra-method "safe_fusion_${pct}pct=safe_fusion_${pct}pct")
      extras+=(--extra-method "svd_${pct}pct=svd_${pct}pct")
      extras+=(--extra-method "weighted_knn_${pct}pct=weighted_knn_${pct}pct")
    done
    "${PY}" scripts/evaluate_pancreas_crossfit_biology.py \
      --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
      --corrupted "${ROOT}/pancreas/fold_0/recorded.h5ad" \
      --coordinates "${ROOT}/pancreas/fold_0/empty_coordinates.parquet" \
      --crossfit-dir "${ROOT}/pancreas" --safe-fusion-subdir safe_fusion \
      "${extras[@]}" --output-dir "${OUT}/pancreas_biology" --bootstrap 2000 --seed 1729 \
      --marker-panel "${MARKER_PANEL}"
    ;;
  4)
    unit="${ROOT}/zebrafish"
    methods_for "${unit}"
    "${PY}" scripts/evaluate_trajectory_preservation.py \
      --truth external_data/prepared/zebrafish_trajectory.h5ad --corrupted "${unit}/recorded.h5ad" \
      --splits "${unit}/splits.parquet" "${methods[@]}" \
      --output-dir "${OUT}/trajectory/zebrafish" --bootstrap 2000 --seed 1729
    ;;
  *)
    datasets=(unused unused unused unused unused norman_crispra adamson_crispri dixit_ko papalexi_eccite)
    interventions=(unused unused unused unused unused gain_of_function loss_of_function loss_of_function loss_of_function)
    dois=(unused unused unused unused unused 10.1126/science.aax4438 10.1016/j.cell.2016.11.048 10.1016/j.cell.2016.11.038 10.1038/s41588-021-00778-2)
    dataset="${datasets[SLURM_ARRAY_TASK_ID]}"
    unit="${ROOT}/${dataset}"
    methods_for "${unit}"
    "${PY}" scripts/evaluate_interventional_grn.py \
      --dataset "${dataset}" --intervention "${interventions[SLURM_ARRAY_TASK_ID]}" \
      --truth "external_data/prepared/${dataset}.h5ad" --corrupted "${unit}/recorded.h5ad" \
      --splits "${unit}/splits.parquet" "${methods[@]}" \
      --publication-doi "${dois[SLURM_ARRAY_TASK_ID]}" \
      --output-dir "${OUT}/grn/${dataset}" --bootstrap 2000 --seed 1729
    ;;
esac
echo "elapsed_s=${SECONDS}"
