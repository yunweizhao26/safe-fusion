#!/usr/bin/env bash
#SBATCH --job-name=sf-r4-sex-masked
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-r4-sex-masked-%A_%a.out
#SBATCH --error=logs/slurm-r4-sex-masked-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUT2=artifacts/paper_evidence/review_round4/transductive_references/sex_zeros
units=(pancreas_0 pancreas_1 pancreas_2 colon_0 colon_1 colon_2)
unit="${units[SLURM_ARRAY_TASK_ID]}"
PANCREAS_TRUTH=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/preprocessed.h5ad
if [[ "${unit}" == pancreas_* ]]; then
  fold="${unit##pancreas_}"
  SRC="artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit/fold_${fold}"
  EXIST="artifacts/paper_evidence/review_round2/fusion_value/transductive/${unit}"
  truth="${SRC}/prepared.h5ad"
  fit_args=(--fit-split development)
else
  fold="${unit##colon_}"
  SRC="artifacts/paper_evidence/review_round3/colon_crossfit/fold_${fold}"
  EXIST="artifacts/paper_evidence/review_round3/colon_crossfit/fusion_value/transductive/colon_${fold}"
  truth="${SRC}/prepared.h5ad"
  fit_count=$(.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('artifacts/paper_evidence/review_round3/colon_crossfit/fold_${fold}/splits.parquet')['split'].isin(['development','validation']).sum())")
  fit_args=(--fit-split validation --fit-cells "$fit_count")
fi
ROOT="${OUT2}/masked_donor/${unit}"
PY=.venv/bin/python
common=(--input "${SRC}/corrupted.h5ad" --coordinates "${SRC}/coordinates.parquet" --splits "${SRC}/splits.parquet" --seed 1729)

case "${STAGE:?STAGE must be donor_teachers, stack or select}" in
  donor_teachers)
    "${PY}" scripts/run_leakage_safe_method.py --method graph_smooth --transductive --condition-column donor \
      "${common[@]}" --output "${ROOT}/graph_smooth_donor"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${SRC}/corrupted.h5ad" --coordinates "${SRC}/coordinates.parquet" --splits "${SRC}/splits.parquet" \
      --output "${ROOT}/scvi_standard_donor" --epochs 200 --seed 1729 --condition-column donor
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/scvi_standard_donor" --corrupted "${SRC}/corrupted.h5ad" \
      --output "${ROOT}/scvi_donor"
    ;;
  stack)
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${ROOT}/safe_fusion_donor" \
      --teacher-contract "${EXIST}/gene_median" --teacher-contract "${EXIST}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth_donor" --teacher-contract "${EXIST}/magic" \
      --teacher-contract "${ROOT}/scvi_donor"
    ;;
  select)
    if [[ ${VARIANT:-both} != donor ]]; then
    plain=(--teacher-contract "${EXIST}/gene_median" --teacher-contract "${EXIST}/svd_impute" \
      --teacher-contract "${EXIST}/graph_smooth" --teacher-contract "${EXIST}/magic" --teacher-contract "${EXIST}/scvi")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${SRC}/corrupted.h5ad" --truth "${truth}" \
      --coordinates "${SRC}/coordinates.parquet" --splits "${SRC}/splits.parquet" \
      --fusion-contract "${EXIST}/safe_fusion" "${plain[@]}" \
      --output-dir "${ROOT}/selector" "${fit_args[@]}" --architecture mlp --budget-mode apply_topk \
      --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729
    fi
    if [[ ${VARIANT:-both} != plain ]]; then
    donor=(--teacher-contract "${EXIST}/gene_median" --teacher-contract "${EXIST}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth_donor" --teacher-contract "${EXIST}/magic" --teacher-contract "${ROOT}/scvi_donor")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${SRC}/corrupted.h5ad" --truth "${truth}" \
      --coordinates "${SRC}/coordinates.parquet" --splits "${SRC}/splits.parquet" \
      --fusion-contract "${ROOT}/safe_fusion_donor" "${donor[@]}" \
      --output-dir "${ROOT}/selector_donor" "${fit_args[@]}" --condition-column donor --condition-feature-source all_cells --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729
    fi
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
