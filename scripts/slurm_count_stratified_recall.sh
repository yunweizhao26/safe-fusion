#!/usr/bin/env bash
#SBATCH --job-name=sf-count-recall
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-count-recall-%x-%A_%a.out
#SBATCH --error=logs/slurm-count-recall-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | stacked | evaluate}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

OUT=artifacts/paper_evidence/review_round2/thinning_transfer/count_stratified_recall
MANIFEST=artifacts/paper_evidence/review_round2/leakage_free/units_manifest.json
KEYS=(pancreas_0 pancreas_1 pancreas_2 colon norman_crispra)
METHODS=(scvi:scVI magic:MAGIC)

case "${stage}" in
  submit)
    mkdir -p logs
    stacked=$(sbatch --parsable --job-name=sf-cr-stk --array=0-9 "${SCRIPT}" stacked)
    evaluate=$(sbatch --parsable --job-name=sf-cr-eval --time=01:00:00 --dependency=afterok:${stacked} "${SCRIPT}" evaluate)
    echo "stacked=${stacked} evaluate=${evaluate}"
    ;;
  stacked)
    key="${KEYS[SLURM_ARRAY_TASK_ID / 2]}"
    spec="${METHODS[SLURM_ARRAY_TASK_ID % 2]}"
    read -r corrupted coordinates splits fit_split unit_column contract < <(.venv/bin/python -c '
import json, sys
unit = next(u for u in json.load(open(sys.argv[1])) if u["key"] == sys.argv[2])
print(unit["corrupted"], unit["coordinates"], unit["splits"], unit["fit_split"], unit["unit_column"], unit["contracts"][sys.argv[3]])
' "${MANIFEST}" "${key}" "${spec##*:}")
    .venv/bin/python scripts/stacked_selector_scores.py --corrupted "${corrupted}" --coordinates "${coordinates}" \
      --splits "${splits}" --fit-split "${fit_split}" --unit-column "${unit_column}" \
      --contract "${contract}" --name "${spec##*:}" \
      --output-dir "${OUT}/stacked/${key}/${spec%%:*}" --seed 1729
    echo "elapsed_s=${SECONDS}"
    ;;
  evaluate)
    .venv/bin/python scripts/evaluate_count_stratified_recall.py --units-manifest "${MANIFEST}" \
      --stacked-root "${OUT}/stacked" --output-dir "${OUT}" --fraction 0.05 --draws 2000 --seed 1729
    echo "elapsed_s=${SECONDS}"
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
