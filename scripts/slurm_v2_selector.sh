#!/usr/bin/env bash
#SBATCH --job-name=sf-v2-selector
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-v2-selector-%x-%A_%a.out
#SBATCH --error=logs/slurm-v2-selector-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

PY=.venv/bin/python
V2=artifacts/paper_evidence/review_round3/selector_v2
MANIFEST=artifacts/paper_evidence/review_round2/leakage_free/units_manifest.json
COLON_MANIFEST=artifacts/paper_evidence/review_round3/colon_crossfit/units_manifest.json
UNITS=(pancreas_0 pancreas_1 pancreas_2 colon norman_crispra colon_0 colon_1 colon_2)
DESIGNS=(entry molecule:0.1 molecule:0.2 entry_molecule:0.1 entry_molecule:0.2)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
FEATURES=(base detection)
task="${SLURM_ARRAY_TASK_ID:-0}"

design_parts() {
  design="${1%%:*}"
  rho=0
  slug="${design}"
  if [[ "$1" == *:* ]]; then
    rho="${1##*:}"
    slug="${design}_${rho/./p}"
  fi
}

chain_paths() {
  unit="${UNITS[$1 / 5]}"
  design_parts "${DESIGNS[$1 % 5]}"
  chain="${V2}/inner/${unit}/${slug}"
  input="${chain}/input/input.h5ad"
  coordinates="${chain}/input/coordinates.parquet"
  splits="${chain}/input/splits.parquet"
  teachers="${chain}/teachers"
  fit_split=development
  manifest="${MANIFEST}"
  [[ "${unit}" == colon* ]] && fit_split=validation
  [[ "${unit}" == colon_* ]] && manifest="${COLON_MANIFEST}"
  return 0
}

case "${stage}" in
  submit_inner)
    mkdir -p logs
    inputs=$(sbatch --parsable -J v2-inner-inputs --array=0-24 --cpus-per-task=2 --mem=24G --time=00:30:00 "${SCRIPT}" inner_inputs)
    cpu=$(sbatch --parsable -J v2-inner-cpu --array=0-99 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" inner_cpu)
    gpu=$(sbatch --parsable -J v2-inner-gpu --array=0-24 --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" inner_gpu)
    selector=$(sbatch --parsable -J v2-inner-sel --array=0-49 -p cs --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" inner_selector)
    evaluate=$(sbatch --parsable -J v2-inner-eval --cpus-per-task=2 --mem=8G --time=00:20:00 --dependency=afterok:${selector} "${SCRIPT}" inner_evaluate)
    echo "inner_inputs=${inputs} inner_cpu=${cpu} inner_gpu=${gpu} inner_selector=${selector} inner_evaluate=${evaluate}"
    ;;
  submit_inner_colon_crossfit)
    mkdir -p logs
    inputs=$(sbatch --parsable -J v2-inner-inputs --array=25-39 --cpus-per-task=2 --mem=24G --time=00:30:00 "${SCRIPT}" inner_inputs)
    cpu=$(sbatch --parsable -J v2-inner-cpu --array=100-159 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" inner_cpu)
    gpu=$(sbatch --parsable -J v2-inner-gpu --array=25-39 --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" inner_gpu)
    selector=$(sbatch --parsable -J v2-inner-sel --array=50-79 -p cs --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" inner_selector)
    evaluate=$(sbatch --parsable -J v2-inner-eval --cpus-per-task=2 --mem=8G --time=00:20:00 --dependency=afterok:${selector} "${SCRIPT}" inner_evaluate)
    echo "inner_inputs=${inputs} inner_cpu=${cpu} inner_gpu=${gpu} inner_selector=${selector} inner_evaluate=${evaluate}"
    ;;
  inner_inputs)
    chain_paths "${task}"
    "${PY}" scripts/v2_selector_inputs.py inner --units-manifest "${manifest}" --unit "${unit}" \
      --design "${design}" --rho "${rho}" --seed 1729 --output-dir "${chain}/input"
    ;;
  inner_cpu)
    chain_paths $((task / 4))
    method="${CPU_TEACHERS[task % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${teachers}/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${teachers}/${method}" --seed 1729
    fi
    ;;
  inner_gpu)
    chain_paths "${task}"
    .conda-scvi-current/bin/python scripts/v2_selector_scvi_teacher.py \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${teachers}/scvi_inductive" --seed 1729
    ;;
  inner_selector)
    chain_paths $((task / 2))
    features="${FEATURES[task % 2]}"
    "${PY}" scripts/v2_selector_fit.py --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --teachers-root "${teachers}" --fit-split "${fit_split}" --features "${features}" \
      --evaluation-column evaluation --output-dir "${chain}/selector_${features}" --seed 1729
    ;;
  inner_evaluate)
    "${PY}" scripts/v2_selector_inner_evaluate.py --root "${V2}/inner" --colon-units colon_0 colon_1 colon_2 \
      --output-dir "${V2}/inner/evaluation_colon_crossfit"
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
echo "elapsed_s=${SECONDS}"
