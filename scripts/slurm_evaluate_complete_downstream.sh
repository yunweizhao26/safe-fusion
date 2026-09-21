#!/usr/bin/env bash
#SBATCH --job-name=sf-down-eval
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-8%3
#SBATCH --output=logs/slurm-complete-downstream-eval-%A_%a.out
#SBATCH --error=logs/slurm-complete-downstream-eval-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
CF=artifacts/paper_evidence/pancreas_crossfit
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
COL_METHODS="${COL}/methods/standardized/colon_epithelial/mask_010"
COL_SELECTOR=artifacts/paper_evidence/selector_mlp_biology_range/colon
COL_GENE_MEDIAN=artifacts/paper_evidence/downstream_complete/teachers/colon/gene_median
NORMAN=artifacts/paper_evidence/norman_crispra
NORMAN_SELECTOR=artifacts/paper_evidence/selector_mlp_biology_range_fullteachers/norman_crispra
OUT=artifacts/paper_evidence/downstream_complete

suffix_for_pct() {
  local pct="$1"
  if (( pct == 10 )); then
    echo 0p1
  else
    echo "0p0${pct}"
  fi
}

if (( SLURM_ARRAY_TASK_ID == 0 )); then
  methods=(
    --method "gene_median=${COL_GENE_MEDIAN}"
    --method "svd=${COL_METHODS}/svd_impute"
    --method "weighted_knn=${COL_METHODS}/graph_smooth"
  )
  for pct in $(seq 1 10); do
    suffix="$(suffix_for_pct "${pct}")"
    methods+=(--method "safe_fusion_${pct}pct=${COL_SELECTOR}/safe_fusion_calibrated_mlp_topk_${suffix}")
  done
  "${PY}" scripts/evaluate_unsupervised_clustering.py \
    --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" \
    --corrupted "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
    --splits "${COL}/data/colon_epithelial/splits.parquet" \
    "${methods[@]}" --label-column cell_type --unit-column donor \
    --output-dir "${OUT}/clustering/colon" --cluster-seeds 20 --bootstrap 2000 --seed 1729

elif (( SLURM_ARRAY_TASK_ID == 1 )); then
  fold_outputs=()
  for fold in 0 1 2; do
    methods=(
      --method "gene_median=${CF}/fold_${fold}/gene_median"
      --method "svd=${CF}/fold_${fold}/svd_impute"
      --method "weighted_knn=${CF}/fold_${fold}/graph_smooth"
    )
    for pct in $(seq 1 10); do
      suffix="$(suffix_for_pct "${pct}")"
      methods+=(--method "safe_fusion_${pct}pct=${CF}/fold_${fold}/selector_mlp_biology_range_fullteachers/safe_fusion_calibrated_mlp_topk_${suffix}")
    done
    fold_output="${OUT}/clustering/pancreas/fold_${fold}"
    fold_outputs+=(--fold "${fold_output}")
    "${PY}" scripts/evaluate_unsupervised_clustering.py \
      --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
      --corrupted "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
      --splits "${CF}/fold_${fold}/splits.parquet" \
      "${methods[@]}" --label-column cell_type --unit-column donor \
      --output-dir "${fold_output}" --cluster-seeds 20 --bootstrap 500 --seed "$((1729 + fold))"
  done
  "${PY}" scripts/combine_clustering_folds.py \
    "${fold_outputs[@]}" --output-dir "${OUT}/clustering/pancreas/combined" \
    --bootstrap 2000 --seed 1729

elif (( SLURM_ARRAY_TASK_ID == 2 )); then
  methods=(
    --method "gene_median=${COL_GENE_MEDIAN}"
    --method "svd=${COL_METHODS}/svd_impute"
    --method "graph_smooth=${COL_METHODS}/graph_smooth"
  )
  for pct in $(seq 1 10); do
    suffix="$(suffix_for_pct "${pct}")"
    methods+=(--method "safe_fusion_${pct}pct=${COL_SELECTOR}/safe_fusion_calibrated_mlp_topk_${suffix}")
  done
  "${PY}" scripts/evaluate_colon_donor_biology.py \
    --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" \
    --corrupted "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
    --coordinates "${COL}/data/colon_epithelial/coordinates/mask_010.parquet" \
    --splits "${COL}/data/colon_epithelial/splits.parquet" \
    "${methods[@]}" --output-dir "${OUT}/markers/colon" --bootstrap 2000 --seed 1729

elif (( SLURM_ARRAY_TASK_ID == 3 )); then
  extras=(
    --extra-method "gene_median=gene_median"
    --extra-method "svd=svd_impute"
  )
  for pct in $(seq 1 10); do
    suffix="$(suffix_for_pct "${pct}")"
    extras+=(--extra-method "safe_fusion_${pct}pct=selector_mlp_biology_range_fullteachers/safe_fusion_calibrated_mlp_topk_${suffix}")
  done
  "${PY}" scripts/evaluate_pancreas_crossfit_biology.py \
    --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
    --corrupted "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
    --coordinates "${PAN}/data/pancreas_islets/coordinates/mask_010.parquet" \
    --crossfit-dir "${CF}" --safe-fusion-subdir safe_fusion_b32 \
    "${extras[@]}" --output-dir "${OUT}/pancreas_biology" --bootstrap 2000 --seed 1729

elif (( SLURM_ARRAY_TASK_ID == 4 )); then
  root=artifacts/external_trajectory/zebrafish
  methods=(
    --method "gene_median=${root}/gene_median"
    --method "svd=${root}/svd_impute"
    --method "weighted_knn=${root}/graph_smooth"
    --method "safe_fusion_dense=${root}/safe_fusion"
  )
  for pct in $(seq 1 10); do
    suffix="$(suffix_for_pct "${pct}")"
    methods+=(--method "safe_fusion_${pct}pct=${root}/selector_mlp_biology_range/safe_fusion_calibrated_mlp_topk_${suffix}")
  done
  "${PY}" scripts/evaluate_trajectory_preservation.py \
    --truth external_data/prepared/zebrafish_trajectory.h5ad \
    --corrupted "${root}/corrupted.h5ad" --splits "${root}/splits.parquet" \
    "${methods[@]}" --output-dir "${OUT}/trajectory/zebrafish" --bootstrap 2000 --seed 1729

else
  case "${SLURM_ARRAY_TASK_ID}" in
    5)
      dataset=norman_crispra
      intervention=gain_of_function
      root="${NORMAN}"
      truth=external_data/prepared/norman_crispra.h5ad
      selector="${NORMAN_SELECTOR}"
      gene_median="${root}/methods/gene_median"
      svd="${root}/methods/svd_impute"
      graph="${root}/methods/graph_smooth"
      fusion="${root}/methods/safe_fusion"
      doi=10.1126/science.aax4438
      ;;
    6|7|8)
      datasets=(unused unused unused unused unused unused adamson_crispri dixit_ko papalexi_eccite)
      interventions=(unused unused unused unused unused unused loss_of_function loss_of_function loss_of_function)
      dois=(unused unused unused unused unused unused 10.1016/j.cell.2016.11.048 10.1016/j.cell.2016.11.038 10.1038/s41588-021-00778-2)
      dataset="${datasets[SLURM_ARRAY_TASK_ID]}"
      intervention="${interventions[SLURM_ARRAY_TASK_ID]}"
      doi="${dois[SLURM_ARRAY_TASK_ID]}"
      root="artifacts/external_perturbseq/${dataset}"
      truth="external_data/prepared/${dataset}.h5ad"
      selector="${root}/selector_mlp_biology_range"
      gene_median="${root}/gene_median"
      svd="${root}/svd_impute"
      graph="${root}/graph_smooth"
      fusion="${root}/safe_fusion"
      ;;
  esac
  methods=(
    --method "gene_median=${gene_median}"
    --method "svd=${svd}"
    --method "weighted_knn=${graph}"
    --method "safe_fusion_dense=${fusion}"
  )
  for pct in $(seq 1 10); do
    suffix="$(suffix_for_pct "${pct}")"
    methods+=(--method "safe_fusion_${pct}pct=${selector}/safe_fusion_calibrated_mlp_topk_${suffix}")
  done
  "${PY}" scripts/evaluate_interventional_grn.py \
    --dataset "${dataset}" --intervention "${intervention}" \
    --truth "${truth}" --corrupted "${root}/corrupted.h5ad" --splits "${root}/splits.parquet" \
    "${methods[@]}" --publication-doi "${doi}" \
    --output-dir "${OUT}/grn/${dataset}" --bootstrap 2000 --seed 1729
fi
