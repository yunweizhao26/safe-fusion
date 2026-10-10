#!/usr/bin/env bash
#SBATCH --job-name=sf-r3-comparators-eval
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-r3-comparators-eval-%x-%A_%a.out
#SBATCH --error=logs/slurm-r3-comparators-eval-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
source scripts/standard_imputers/r3_units.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PY=.venv/bin/python
METHODS=(dca dca_zinb scimpute screcover enimpute scvi_zinb)
NAMES=("DCA" "DCA (ZINB)" "scImpute" "scRecover" "EnImpute" "scVI (ZINB)")
SLUGS=(dca dca_zinb scimpute screcover enimpute scvi_zinb)
PROBABILITY=(dca_zinb scimpute screcover scvi_zinb)
FILL_UNITS=(deploy_pancreas_0 deploy_pancreas_1 deploy_pancreas_2 deploy_colon deploy_zebrafish deploy_norman_crispra)
CF=artifacts/paper_evidence/review_round3/colon_crossfit
task="${SLURM_ARRAY_TASK_ID:-0}"

masked_unit() {
  case "$1" in
    pancreas_*) key="$1"; fit_split=development; unit_column=donor ;;
    norman_crispra) key="$1"; fit_split=development; unit_column=target ;;
    colon) key=colon; fit_split=validation; unit_column=donor ;;
    colon_cf_*) key="colon_${1##*_}"; fit_split=validation; unit_column=donor ;;
  esac
}

case "${STAGE:?STAGE must be units, stacked, fills, masked, masked_cf or known_zeros}" in
  units)
    "${PY}" scripts/standard_imputers/write_r3_units.py --root "${R3_ROOT}"
    ;;
  stacked)
    unit="${R3_MASKED[task / ${#METHODS[@]}]}"
    method_index=$((task % ${#METHODS[@]}))
    r3_unit_paths "${unit}"
    masked_unit "${unit}"
    "${PY}" scripts/stacked_selector_scores.py --corrupted "${corrupted}" --coordinates "${coordinates}" \
      --splits "${splits}" --fit-split "${fit_split}" --unit-column "${unit_column}" \
      --contract "${R3_ROOT}/fits/${METHODS[method_index]}/${unit}" --name "${NAMES[method_index]}" \
      --output-dir "${R3_ROOT}/stacked/${key}/${SLUGS[method_index]}" --seed 1729
    ;;
  fills)
    unit="${FILL_UNITS[task]}"
    r3_unit_paths "${unit}"
    for method in "${METHODS[@]}"; do
      contract="${R3_ROOT}/fits/${method}/${unit}"
      [[ -f "${contract}/mean.npy" ]] || { echo "missing fit ${contract}" >&2; continue; }

      [[ -f "${R3_ROOT}/deployment_fills/${unit#deploy_}/${method}_topk_0p1/mean.npy" ]] && continue
      "${PY}" scripts/standard_imputers/r3_deployment_fills.py --hybrid "${corrupted}" --recorded "${recorded}" \
        --splits "${splits}" --contract "${contract}" --name "${method}" --ranking value \
        --output-root "${R3_ROOT}/deployment_fills/${unit#deploy_}"
      if [[ " ${PROBABILITY[*]} " == *" ${method} "* ]]; then
        "${PY}" scripts/standard_imputers/r3_deployment_fills.py --hybrid "${corrupted}" --recorded "${recorded}" \
          --splits "${splits}" --contract "${contract}" --name "${method}_dropout" --ranking dropout \
          --output-root "${R3_ROOT}/deployment_fills/${unit#deploy_}"
      fi
    done
    ;;
  masked)
    "${PY}" scripts/standard_imputers/evaluate_r3_masked.py --units-manifest "${R3_ROOT}/units_manifest.json" \
      --unit-keys pancreas_0 pancreas_1 pancreas_2 colon norman_crispra \
      --output-dir "${R3_ROOT}/evaluation/masked"
    ;;
  masked_cf)
    "${PY}" scripts/standard_imputers/evaluate_r3_masked.py --units-manifest "${R3_ROOT}/colon_crossfit_units.json" \
      --unit-keys colon_0 colon_1 colon_2 --dataset-label "Colon cross-fit" \
      --fusion-selector-root "${CF}/fusion_value/selectors" --probability-root "${CF}/fusion_value/nonzero_probability" \
      --output-dir "${R3_ROOT}/evaluation/masked_colon_crossfit"
    ;;
  known_zeros)
    "${PY}" scripts/standard_imputers/evaluate_r3_known_zeros.py --output-dir "${R3_ROOT}/evaluation/known_zeros"
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
