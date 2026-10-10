#!/usr/bin/env bash
#SBATCH --job-name=sf-fusion-value
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-fusion-value-%A_%a.out
#SBATCH --error=logs/slurm-fusion-value-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

ROOT="${ROOT:-artifacts/paper_evidence/review_round2/fusion_value}"
TRANSDUCTIVE="${ROOT}/transductive"
MANIFEST="${MANIFEST:-artifacts/paper_evidence/review_round2/leakage_free/units_manifest.json}"
STAGE="${STAGE:?set STAGE to teachers, stack, selectors, probability or evaluate}"

units=(${UNITS:-pancreas_0 pancreas_1 pancreas_2 colon norman_crispra})

unit_paths() {
  read -r input coordinates splits magic scvi saver < <(.venv/bin/python scripts/fusion_value_selectors.py \
    --units-manifest "${MANIFEST}" --unit "$1" --unit-keys "${units[@]}" --unit-paths)
}

case "${STAGE}" in
  teachers)
    teachers=(gene_median svd_impute graph_smooth magic scvi)
    key="${units[SLURM_ARRAY_TASK_ID / 5]}"
    teacher="${teachers[SLURM_ARRAY_TASK_ID % 5]}"
    unit_paths "${key}"
    output="${TRANSDUCTIVE}/${key}/${teacher}"
    case "${teacher}" in
      magic|scvi)
        .venv/bin/python scripts/count_scale_contract.py \
          --contract "${!teacher}" --corrupted "${input}" --output "${output}"
        ;;
      *)
        .venv/bin/python scripts/run_leakage_safe_method.py --method "${teacher}" --transductive \
          --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
          --output "${output}" --seed 1729
        ;;
    esac
    ;;
  stack)
    key="${units[SLURM_ARRAY_TASK_ID]}"
    unit_paths "${key}"
    teacher_args=()
    for teacher in gene_median svd_impute graph_smooth magic scvi; do
      teacher_args+=(--teacher-contract "${TRANSDUCTIVE}/${key}/${teacher}")
    done
    .venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion --transductive \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${TRANSDUCTIVE}/${key}/safe_fusion" --seed 1729 "${teacher_args[@]}"
    ;;
  selectors)

    .venv/bin/python scripts/fusion_value_selectors.py --units-manifest "${MANIFEST}" \
      --unit-keys "${units[@]}" --output-root "${ROOT}/selectors" --transductive-root "${TRANSDUCTIVE}" \
      --array-index "${SLURM_ARRAY_TASK_ID}" \
      --families ${FAMILIES:-fusion leave_one_out architecture stacked} --seed 1729
    ;;
  probability)
    methods=(scvi saver)
    key="${units[SLURM_ARRAY_TASK_ID / 2]}"
    method="${methods[SLURM_ARRAY_TASK_ID % 2]}"
    unit_paths "${key}"
    .conda-scvi-current/bin/python scripts/nonzero_probability.py --method "${method}" \
      --contract "${!method}" --corrupted "${input}" --output "${ROOT}/nonzero_probability/${method}/${key}"
    ;;
  evaluate)
    .venv/bin/python scripts/fusion_value_bootstrap.py --units-manifest "${MANIFEST}" --seed 1729
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
