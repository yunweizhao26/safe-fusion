#!/usr/bin/env bash
#SBATCH --job-name=sf-r4-sex-deploy
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-r4-sex-deploy-%A_%a.out
#SBATCH --error=logs/slurm-r4-sex-deploy-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUT2=artifacts/paper_evidence/review_round4/transductive_references/sex_zeros
units=(pancreas_0 pancreas_1 pancreas_2 colon_0 colon_1 colon_2)
unit="${units[SLURM_ARRAY_TASK_ID]}"
ROOT="${OUT2}/deployment_rebuilt/${unit}"
PANCREAS_TRUTH=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/preprocessed.h5ad
if [[ "${unit}" == pancreas_* ]]; then
  truth="artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit/fold_${unit##*_}/prepared.h5ad"
  fit_args=(--fit-split development)
else
  fold="${unit##colon_}"
  truth="artifacts/paper_evidence/review_round3/colon_crossfit/fold_${fold}/prepared.h5ad"
  fit_count=$(.venv/bin/python -c "import pandas as pd; print(pd.read_parquet('artifacts/paper_evidence/review_round3/colon_crossfit/fold_${fold}/splits.parquet')['split'].isin(['development','validation']).sum())")
  fit_args=(--fit-split validation --fit-cells "$fit_count")
fi
PY=.venv/bin/python
common=(--input "${ROOT}/hybrid.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" --seed 1729)
SEX_GENES=(ENSG00000229807 ENSG00000129824)

case "${STAGE:?STAGE must be magic, scvi, scvi_donor, teachers, stack or select}" in
  magic)
    .conda-magic-current/bin/python scripts/run_magic_baseline.py \
      --corrupted "${ROOT}/hybrid.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --output "${ROOT}/magic_standard" --n-jobs "${SLURM_CPUS_PER_TASK}" --seed 1729
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/magic_standard" --corrupted "${ROOT}/hybrid.h5ad" \
      --output "${ROOT}/magic"
    ;;
  scvi)
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${ROOT}/hybrid.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --output "${ROOT}/scvi_standard" --epochs 200 --seed 1729
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/scvi_standard" --corrupted "${ROOT}/hybrid.h5ad" \
      --output "${ROOT}/scvi"
    ;;
  scvi_donor)
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${ROOT}/hybrid.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --output "${ROOT}/scvi_standard_donor" --epochs 200 --seed 1729 --condition-column donor
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/scvi_standard_donor" --corrupted "${ROOT}/hybrid.h5ad" \
      --output "${ROOT}/scvi_donor"
    ;;
  teachers)
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" --transductive "${common[@]}" --output "${ROOT}/${method}"
    done
    "${PY}" scripts/run_leakage_safe_method.py --method graph_smooth --transductive --condition-column donor \
      "${common[@]}" --output "${ROOT}/graph_smooth_donor"
    ;;
  stack)
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${ROOT}/safe_fusion" \
      --teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth" --teacher-contract "${ROOT}/magic" --teacher-contract "${ROOT}/scvi"
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${ROOT}/safe_fusion_donor" \
      --teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth_donor" --teacher-contract "${ROOT}/magic" --teacher-contract "${ROOT}/scvi_donor"
    ;;
  select)
    plain=(--teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth" --teacher-contract "${ROOT}/magic" --teacher-contract "${ROOT}/scvi")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/hybrid.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/safe_fusion" "${plain[@]}" \
      --output-dir "${ROOT}/selector" "${fit_args[@]}" --architecture mlp --budget-mode apply_topk \
      --budgets 0.01 0.05 0.10 --curve-points 2 --score-gene "${SEX_GENES[@]}" --seed 1729
    donor=(--teacher-contract "${ROOT}/gene_median" --teacher-contract "${ROOT}/svd_impute" \
      --teacher-contract "${ROOT}/graph_smooth_donor" --teacher-contract "${ROOT}/magic" --teacher-contract "${ROOT}/scvi_donor")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/hybrid.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/safe_fusion_donor" "${donor[@]}" \
      --output-dir "${ROOT}/selector_donor" "${fit_args[@]}" --condition-column donor --condition-feature-source all_cells --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.05 0.10 --curve-points 2 --score-gene "${SEX_GENES[@]}" --seed 1729
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
