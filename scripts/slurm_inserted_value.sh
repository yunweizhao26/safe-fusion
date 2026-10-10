#!/usr/bin/env bash
#SBATCH --job-name=sf-inserted-value
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-inserted-value-%x-%j.out
#SBATCH --error=logs/slurm-inserted-value-%x-%j.err

set -euo pipefail

stage="${1:?stage: submit | prepare | scvi | fill | evaluate}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

PY=.venv/bin/python
NORMAN=artifacts/paper_evidence/review_round2/leakage_free/norman_crispra
OUT=artifacts/paper_evidence/review_round2/inserted_value/deployment/norman_crispra
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)
FRACTIONS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

case "${stage}" in
  submit)
    mkdir -p logs
    prep=$(sbatch --parsable --job-name=sf-iv-prep "${SCRIPT}" prepare)
    scvi=$(sbatch --parsable --job-name=sf-iv-scvi --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${prep} "${SCRIPT}" scvi)
    fill=$(sbatch --parsable --job-name=sf-iv-fill --dependency=afterok:${scvi} "${SCRIPT}" fill)
    eval=$(sbatch --parsable --job-name=sf-iv-eval --cpus-per-task=4 --time=01:00:00 \
      --dependency=afterok:${fill}${2:+:$2} "${SCRIPT}" evaluate)
    echo "prepare=${prep} scvi=${scvi} fill=${fill} evaluate=${eval}"
    ;;
  prepare)
    "${PY}" scripts/build_deployment_inputs.py --truth "${NORMAN}/prepared.h5ad" --corrupted "${NORMAN}/corrupted.h5ad" \
      --coordinates "${NORMAN}/coordinates.parquet" --splits "${NORMAN}/splits.parquet" --output-dir "${OUT}"
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" --input "${OUT}/hybrid.h5ad" \
        --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
        --output "${OUT}/methods/${method}" --seed 1729
    done
    .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic --input "${OUT}/hybrid.h5ad" \
      --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
      --output "${OUT}/methods/magic_inductive" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
    ;;
  scvi)
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi --input "${OUT}/hybrid.h5ad" \
      --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
      --output "${OUT}/methods/scvi_inductive" --seed 1729
    ;;
  fill)
    teacher_args=()
    for name in "${TEACHERS[@]}"; do teacher_args+=(--teacher-contract "${OUT}/methods/${name}"); done
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --input "${OUT}/hybrid.h5ad" \
      --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
      --output "${OUT}/methods/safe_fusion" --seed 1729 "${teacher_args[@]}"
    "${PY}" scripts/calibrated_selective_fill.py --corrupted "${OUT}/hybrid.h5ad" --truth "${NORMAN}/prepared.h5ad" \
      --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
      --fusion-contract "${OUT}/methods/safe_fusion" "${teacher_args[@]}" --output-dir "${OUT}/selector" \
      --fit-split development --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
      --curve-points 2 --seed 1729 > /dev/null
    for spec in "svd_impute svd" "graph_smooth weighted_knn"; do
      set -- ${spec}
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${OUT}/hybrid.h5ad" --splits "${OUT}/splits.parquet" \
        --method-contract "${OUT}/methods/$1" --method-name "$2" --output-root "${OUT}/matched" --fractions "${FRACTIONS[@]}"
    done
    ;;
  evaluate)
    "${PY}" scripts/evaluate_inserted_value.py all --draws 2000 --seed 1729
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
echo "elapsed_s=${SECONDS}"
