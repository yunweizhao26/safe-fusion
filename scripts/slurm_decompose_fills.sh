#!/usr/bin/env bash
#SBATCH --job-name=sf-decompose
#SBATCH --account=torch_pr_634_general
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-8%3
#SBATCH --output=logs/slurm-decompose-fills-%A_%a.out
#SBATCH --error=logs/slurm-decompose-fills-%A_%a.err

# Split each sparse fill of the masked benchmark into the selected masked
# positives and the selected recorded zeros, and evaluate both parts with the
# downstream evaluators. Task = dataset (colon, pancreas, zebrafish) x
# source (Safe Fusion, SVD, weighted kNN).
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
# Fill locations; override to decompose another set of outputs.
CF="${CF:-artifacts/paper_evidence/pancreas_crossfit}"
COL_SELECTOR="${COL_SELECTOR:-artifacts/paper_evidence/selector_mlp_biology_range/colon}"
COL_MATCHED="${COL_MATCHED:-artifacts/paper_evidence/matched_fraction/colon}"
ZF_FILLS="${ZF_FILLS:-artifacts/external_trajectory/zebrafish}"
ZF=artifacts/external_trajectory/zebrafish
ROOT="${DECOMP_ROOT:-artifacts/paper_evidence/downstream_decomposition}"
PCTS="${PCTS:-1 2 3 4 5 6 7 8 9 10}"
MARKER_PANEL="${MARKER_PANEL:-source}"

datasets=(colon pancreas zebrafish)
sources=(safe_fusion svd weighted_knn)
dataset="${datasets[SLURM_ARRAY_TASK_ID / 3]}"
source_method="${sources[SLURM_ARRAY_TASK_ID % 3]}"

suffix_for_pct() {
  if (( $1 == 10 )); then echo 0p1; else echo "0p0$1"; fi
}

# source_dir <fill root> <pct>: the sparse contract of this task's source.
source_dir() {
  local selector_root="$1" matched_root="$2" pct="$3" suffix
  suffix="$(suffix_for_pct "${pct}")"
  case "${source_method}" in
    safe_fusion) echo "${selector_root}/safe_fusion_calibrated_mlp_topk_${suffix}" ;;
    svd) echo "${matched_root}/svd_topk_${suffix}" ;;
    weighted_knn) echo "${matched_root}/weighted_knn_topk_${suffix}" ;;
  esac
}

# decompose <corrupted> <coordinates> <splits> <selector root> <matched root> <output root>
# Sets the global array "methods" to the full fills and both parts.
decompose() {
  local corrupted="$1" coordinates="$2" splits="$3" selector_root="$4" matched_root="$5" out="$6"
  local sources_args=() pct name
  methods=()
  for pct in ${PCTS}; do
    name="${source_method}_${pct}pct"
    sources_args+=(--source "${name}=$(source_dir "${selector_root}" "${matched_root}" "${pct}")")
    methods+=(--method "${name}=$(source_dir "${selector_root}" "${matched_root}" "${pct}")")
    methods+=(--method "${name}__masked_only=${out}/${name}__masked_only")
    methods+=(--method "${name}__zeros_only=${out}/${name}__zeros_only")
  done
  "${PY}" scripts/decompose_fills.py --corrupted "${corrupted}" --coordinates "${coordinates}" \
    --splits "${splits}" "${sources_args[@]}" --output-root "${out}"
}

case "${dataset}" in
  colon)
    out="${ROOT}/colon/${source_method}"
    decompose "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
      "${COL}/data/colon_epithelial/coordinates/mask_010.parquet" \
      "${COL}/data/colon_epithelial/splits.parquet" "${COL_SELECTOR}" "${COL_MATCHED}" "${out}"
    "${PY}" scripts/evaluate_colon_donor_biology.py \
      --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" \
      --corrupted "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
      --coordinates "${COL}/data/colon_epithelial/coordinates/mask_010.parquet" \
      --splits "${COL}/data/colon_epithelial/splits.parquet" \
      "${methods[@]}" --output-dir "${out}/evaluation/markers" --bootstrap 2000 --seed 1729 \
      --marker-panel "${MARKER_PANEL}"
    "${PY}" scripts/evaluate_unsupervised_clustering.py \
      --truth "${COL}/data/colon_epithelial/preprocessed.h5ad" \
      --corrupted "${COL}/data/colon_epithelial/corrupted/mask_010.h5ad" \
      --splits "${COL}/data/colon_epithelial/splits.parquet" \
      "${methods[@]}" --label-column cell_type --unit-column donor \
      --output-dir "${out}/evaluation/clustering" --cluster-seeds 20 --bootstrap 2000 --seed 1729
    ;;
  pancreas)
    extras=()
    fold_outputs=()
    for fold in 0 1 2; do
      fold_root="${ROOT}/pancreas/${source_method}/fold_${fold}"
      decompose "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
        "${PAN}/data/pancreas_islets/coordinates/mask_010.parquet" \
        "${CF}/fold_${fold}/splits.parquet" \
        "${CF}/fold_${fold}/selector_mlp_biology_range_fullteachers" \
        "${CF}/fold_${fold}/matched_fraction" "${fold_root}"
      # The cross-fit evaluator reads every method from <crossfit-dir>/fold_k/<subdir>.
      for linked in splits.parquet graph_smooth safe_fusion; do
        ln -sfn "$(realpath "${CF}/fold_${fold}/${linked}")" "${fold_root}/${linked}"
      done
      for pct in ${PCTS}; do
        name="${source_method}_${pct}pct"
        ln -sfn "$(realpath "$(source_dir "${CF}/fold_${fold}/selector_mlp_biology_range_fullteachers" "${CF}/fold_${fold}/matched_fraction" "${pct}")")" "${fold_root}/${name}"
      done
      fold_output="${ROOT}/pancreas/${source_method}/evaluation/clustering/fold_${fold}"
      fold_outputs+=(--fold "${fold_output}")
      "${PY}" scripts/evaluate_unsupervised_clustering.py \
        --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
        --corrupted "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
        --splits "${CF}/fold_${fold}/splits.parquet" \
        "${methods[@]}" --label-column cell_type --unit-column donor \
        --output-dir "${fold_output}" --cluster-seeds 20 --bootstrap 500 --seed "$((1729 + fold))"
    done
    "${PY}" scripts/combine_clustering_folds.py "${fold_outputs[@]}" \
      --output-dir "${ROOT}/pancreas/${source_method}/evaluation/clustering/combined" \
      --bootstrap 2000 --seed 1729
    for pct in ${PCTS}; do
      name="${source_method}_${pct}pct"
      extras+=(--extra-method "${name}=${name}")
      extras+=(--extra-method "${name}__masked_only=${name}__masked_only")
      extras+=(--extra-method "${name}__zeros_only=${name}__zeros_only")
    done
    "${PY}" scripts/evaluate_pancreas_crossfit_biology.py \
      --truth "${PAN}/data/pancreas_islets/preprocessed.h5ad" \
      --corrupted "${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad" \
      --coordinates "${PAN}/data/pancreas_islets/coordinates/mask_010.parquet" \
      --crossfit-dir "${ROOT}/pancreas/${source_method}" --safe-fusion-subdir safe_fusion \
      "${extras[@]}" --output-dir "${ROOT}/pancreas/${source_method}/evaluation/biology" \
      --bootstrap 2000 --seed 1729 --marker-panel "${MARKER_PANEL}"
    ;;
  zebrafish)
    out="${ROOT}/zebrafish/${source_method}"
    decompose "${ZF}/corrupted.h5ad" "${ZF}/coordinates.parquet" "${ZF}/splits.parquet" \
      "${ZF_FILLS}/selector_mlp_biology_range" "${ZF_FILLS}/matched_fraction" "${out}"
    "${PY}" scripts/evaluate_trajectory_preservation.py \
      --truth external_data/prepared/zebrafish_trajectory.h5ad \
      --corrupted "${ZF}/corrupted.h5ad" --splits "${ZF}/splits.parquet" \
      "${methods[@]}" --output-dir "${out}/evaluation/trajectory" --bootstrap 2000 --seed 1729
    ;;
esac
