#!/usr/bin/env bash
#SBATCH --job-name=sf-seed-rep
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-seed-rep-%x-%A_%a.out
#SBATCH --error=logs/slurm-seed-rep-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
CF=artifacts/paper_evidence/pancreas_crossfit
REP=artifacts/paper_evidence/seed_replicates
SEEDS=(1730 1731 1732 1733)
DATASETS=(pancreas colon)
REP_UNITS=(pancreas_0 pancreas_1 pancreas_2 colon)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
task="${SLURM_ARRAY_TASK_ID}"

dataset_paths() {
  case "$1" in
    pancreas) truth="${PAN}/data/pancreas_islets/preprocessed.h5ad"; workflow_splits="${PAN}/data/pancreas_islets/splits.parquet"; unit_column=donor ;;
    colon) truth="${COL}/data/colon_epithelial/preprocessed.h5ad"; workflow_splits="${COL}/data/colon_epithelial/splits.parquet"; unit_column=donor ;;
  esac
}

replicate_paths() {
  case "$1" in
    pancreas_*) dataset=pancreas; splits="${CF}/fold_${1##*_}/splits.parquet"; fit=(--fit-split development) ;;
    colon) dataset=colon; splits="${COL}/data/colon_epithelial/splits.parquet"; fit=(--fit-split validation --fit-cells 3852) ;;
  esac
  data="${REP}/seed_$2/data/${dataset}"
  out="${REP}/seed_$2/$1"
}

case "${STAGE}" in
  mask)
    seed="${SEEDS[task / 2]}"
    dataset_paths "${DATASETS[task % 2]}"
    "${PY}" scripts/make_mask_replicate.py --truth "${truth}" --splits "${workflow_splits}" \
      --unit-column "${unit_column}" --seed "${seed}" \
      --output-dir "${REP}/seed_${seed}/data/${DATASETS[task % 2]}"
    ;;
  teachers)
    seed="${SEEDS[task / 16]}"
    method="${CPU_TEACHERS[task % 4]}"
    replicate_paths "${REP_UNITS[(task % 16) / 4]}" "${seed}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
        --splits "${splits}" --output "${out}/${method}" \
        --seed "${seed}" --n-jobs "${SLURM_CPUS_PER_TASK}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
        --splits "${splits}" --output "${out}/${method}" --seed "${seed}"
    fi
    ;;
  scvi_teacher)
    seed="${SEEDS[task / 4]}"
    replicate_paths "${REP_UNITS[task % 4]}" "${seed}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${splits}" --output "${out}/scvi_inductive" --seed "${seed}"
    ;;
  stack)
    seed="${SEEDS[task / 4]}"
    replicate_paths "${REP_UNITS[task % 4]}" "${seed}"
    mapfile -t teacher_args < <(teacher_contract_args "${out}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${splits}" --output "${out}/$(fused_value_contract "${VALUE_MODEL:-boosted}")" \
      --value-model "${VALUE_MODEL:-boosted}" --seed "${seed}" "${teacher_args[@]}"
    ;;
  selector)
    seed="${SEEDS[task / 4]}"
    replicate_paths "${REP_UNITS[task % 4]}" "${seed}"
    dataset_paths "${dataset}"
    mapfile -t teacher_args < <(teacher_contract_args "${out}")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${data}/corrupted.h5ad" --truth "${truth}" \
      --coordinates "${data}/coordinates.parquet" --splits "${splits}" \
      --fusion-contract "${out}/safe_fusion" \
      "${teacher_args[@]}" \
      --output-dir "${out}/selector" \
      "${fit[@]}" \
      --architecture mlp --budget-mode apply_topk \
      --budgets 0.01 0.05 0.10 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
      --seed "${seed}"
    ;;
  magic)
    seed="${SEEDS[task / 2]}"
    dataset="${DATASETS[task % 2]}"
    dataset_paths "${dataset}"
    data="${REP}/seed_${seed}/data/${dataset}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py \
      --corrupted "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${workflow_splits}" --output "${REP}/seed_${seed}/baselines/magic/${dataset}" \
      --n-jobs "${SLURM_CPUS_PER_TASK}" --seed "${seed}"
    ;;
  scvi)
    seed="${SEEDS[task / 2]}"
    dataset="${DATASETS[task % 2]}"
    dataset_paths "${dataset}"
    data="${REP}/seed_${seed}/data/${dataset}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${workflow_splits}" --output "${REP}/seed_${seed}/baselines/scvi/${dataset}" \
      --epochs 200 --seed "${seed}"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
