#!/usr/bin/env bash
#SBATCH --job-name=sf-thinning
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-thinning-%x-%A_%a.out
#SBATCH --error=logs/slurm-thinning-%x-%A_%a.err

# Binomial-thinning benchmark. The positives are held-out entries that are
# nonzero in the recorded counts and zero after thinning, so their chance of
# becoming zero depends on the count. Every model is fitted on the thinned
# model-fitting cells, exactly as in the masked benchmark: the five teachers,
# the stacked value, and the MLP selector trained on the fitting cells' thinned
# entries against their recorded zeros. MAGIC and scVI are also run as
# comparators on all cells. Outputs go to artifacts/paper_evidence/thinning/.
#
# Submit every stage from the repository root with
#   bash scripts/slurm_thinning_benchmark.sh submit
# Stages: prepare (array 0-2), cpu (0-22), gpu (0-7, on an L40S GPU like every
# scVI fit, because the fitted values differ between GPU models),
# selector (0-4), evaluate. The stage stack (0-4) refits only the fused value
# with VALUE_MODEL: boosted (default) or linear, the linear combination of the
# teachers written to <unit>/safe_fusion_linear.
set -euo pipefail

stage="${1:?stage: submit | prepare | cpu | gpu | stack | selector | evaluate}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
source scripts/unit_paths.sh  # fused_value_contract; the units are defined below
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

T=artifacts/paper_evidence/thinning
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets
CF=artifacts/paper_evidence/pancreas_crossfit
DATASETS=(colon_thinning_050 colon_thinning_025 pancreas_thinning_050)
UNITS=(colon_thinning_050 colon_thinning_025 pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

# Set input, coordinates, splits, truth, methods_root and fit for a unit.
unit_paths() {
  local key="$1"
  case "${key}" in
    colon_thinning_*)
      input="${T}/data/${key}/corrupted.h5ad"
      coordinates="${T}/data/${key}/coordinates.parquet"
      splits="${COL}/splits.parquet"
      truth="${COL}/preprocessed.h5ad"
      methods_root="${T}/${key}"
      fit=(--fit-split validation --fit-cells 3852)
      ;;
    pancreas_thinning_050_*)
      input="${T}/data/pancreas_thinning_050/corrupted.h5ad"
      coordinates="${T}/data/pancreas_thinning_050/coordinates.parquet"
      splits="${CF}/fold_${key##*_}/splits.parquet"
      truth="${PAN}/preprocessed.h5ad"
      methods_root="${T}/${key}"
      fit=(--fit-split development)
      ;;
    *) echo "unknown unit ${key}" >&2; return 2 ;;
  esac
}

# Fit the fused value of the current unit from its five teachers with value model $1.
fit_value() {
  local teacher_args=() name
  for name in "${TEACHERS[@]}"; do teacher_args+=(--teacher-contract "${methods_root}/${name}"); done
  .venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion \
    --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
    --output "${methods_root}/$(fused_value_contract "$1")" --value-model "$1" \
    --seed 1729 "${teacher_args[@]}"
}

# Set input, coordinates and splits for a whole dataset (all-cell comparators).
dataset_paths() {
  case "$1" in
    colon_*) unit_paths "$1" ;;
    pancreas_thinning_050) unit_paths pancreas_thinning_050_0 ;;
  esac
}

case "${stage}" in
  submit)
    mkdir -p logs
    prep=$(sbatch --parsable --job-name=sf-thin-prep --array=0-2 --mem=16G --time=00:30:00 "${SCRIPT}" prepare)
    cpu=$(sbatch --parsable --job-name=sf-thin-cpu --array=0-22 --dependency=afterok:${prep} "${SCRIPT}" cpu)
    gpu=$(sbatch --parsable --job-name=sf-thin-gpu --array=0-7 --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${prep} "${SCRIPT}" gpu)
    sel=$(sbatch --parsable --job-name=sf-thin-sel --array=0-4 --mem=24G --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" selector)
    eval=$(sbatch --parsable --job-name=sf-thin-eval --cpus-per-task=4 --mem=32G --time=01:00:00 --dependency=afterok:${sel} "${SCRIPT}" evaluate)
    echo "prepare=${prep} cpu=${cpu} gpu=${gpu} selector=${sel} evaluate=${eval}"
    ;;
  prepare)
    key="${DATASETS[SLURM_ARRAY_TASK_ID]}"
    case "${key}" in
      colon_thinning_050) source_args=(--truth "${COL}/preprocessed.h5ad" --corrupted "${COL}/corrupted/thinning_050.h5ad") ;;
      colon_thinning_025) source_args=(--truth "${COL}/preprocessed.h5ad" --corrupted "${COL}/corrupted/thinning_025.h5ad") ;;
      pancreas_thinning_050) source_args=(--truth "${PAN}/preprocessed.h5ad" --retained-fraction 0.5 --seed 1729) ;;
    esac
    .venv/bin/python scripts/prepare_thinning_benchmark.py "${source_args[@]}" --output-dir "${T}/data/${key}"
    ;;
  cpu)
    # Tasks 0-19: the four CPU teachers for each unit. Tasks 20-22: MAGIC on
    # all cells of each dataset as a comparator.
    if (( SLURM_ARRAY_TASK_ID >= 20 )); then
      key="${DATASETS[SLURM_ARRAY_TASK_ID - 20]}"
      dataset_paths "${key}"
      .conda-magic-current/bin/python scripts/run_magic_baseline.py \
        --corrupted "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${T}/comparators/${key}/magic" --n-jobs "${OMP_NUM_THREADS}" --seed 1729
      exit 0
    fi
    unit_paths "${UNITS[SLURM_ARRAY_TASK_ID / 4]}"
    method="${CPU_TEACHERS[SLURM_ARRAY_TASK_ID % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${methods_root}/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
    else
      .venv/bin/python scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${methods_root}/${method}" --seed 1729
    fi
    ;;
  gpu)
    # Tasks 0-4: the inductive scVI teacher for each unit. Tasks 5-7: scVI on
    # all cells of each dataset as a comparator.
    if (( SLURM_ARRAY_TASK_ID >= 5 )); then
      key="${DATASETS[SLURM_ARRAY_TASK_ID - 5]}"
      dataset_paths "${key}"
      .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
        --corrupted "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${T}/comparators/${key}/scvi" --epochs 200 --seed 1729
      exit 0
    fi
    unit_paths "${UNITS[SLURM_ARRAY_TASK_ID]}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${methods_root}/scvi_inductive" --seed 1729
    ;;
  stack)
    unit_paths "${UNITS[SLURM_ARRAY_TASK_ID]}"
    fit_value "${VALUE_MODEL:-boosted}"
    ;;
  selector)
    unit_paths "${UNITS[SLURM_ARRAY_TASK_ID]}"
    fit_value boosted
    teacher_args=()
    for name in "${TEACHERS[@]}"; do teacher_args+=(--teacher-contract "${methods_root}/${name}"); done
    .venv/bin/python scripts/calibrated_selective_fill.py \
      --corrupted "${input}" --truth "${truth}" --coordinates "${coordinates}" --splits "${splits}" \
      --fusion-contract "${methods_root}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${methods_root}/selector" "${fit[@]}" \
      --architecture mlp --budget-mode apply_topk --budgets "${BUDGETS[@]}" \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729 > /dev/null
    ;;
  evaluate)
    .venv/bin/python scripts/evaluate_thinning_benchmark.py --root "${T}" --bootstrap 2000 --seed 7
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
