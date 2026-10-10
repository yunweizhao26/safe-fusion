#!/usr/bin/env bash
#SBATCH --job-name=sf-r4-kd-deploy
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-r4-kd-deploy-%A_%a.out
#SBATCH --error=logs/slurm-r4-kd-deploy-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

OUT1=artifacts/paper_evidence/review_round4/transductive_references/knockdown
screens=(adamson_crispri papalexi_eccite norman_crispra)
screen="${screens[SLURM_ARRAY_TASK_ID]}"
if [[ "${screen}" == norman_crispra ]]; then
  ROOT="${OUT1}/norman_rebuilt/deployment"
  truth="${OUT1}/masked/norman_crispra/prepared.h5ad"
  STANDARD_ROOT="${OUT1}/norman_rebuilt/standard_imputers"
else
  ROOT="${OUT1}/deployment/${screen}"
  truth="external_data/prepared/${screen}.h5ad"
  STANDARD_ROOT="${OUT1}/review_root/standard_imputers"
fi
PY=.venv/bin/python
common=(--input "${ROOT}/hybrid.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" --seed 1729)

target_genes() {
  "${PY}" - "${ROOT}/hybrid.h5ad" <<'EOF'
import sys
import anndata as ad
adata = ad.read_h5ad(sys.argv[1], backed="r")
symbols = adata.var["feature_name"].astype(str) if "feature_name" in adata.var else adata.var_names.astype(str)
lookup = dict(zip(symbols, adata.var_names.astype(str)))
for target in sorted(set(adata.obs["target"].astype(str)) - {"none"}):
    if target in lookup:
        print(lookup[target])
EOF
}

standard_dir() {
  if [[ "${screen}" == norman_crispra ]]; then echo "${STANDARD_ROOT}/$1"; else echo "${STANDARD_ROOT}/$1/${screen}"; fi
}

case "${STAGE:?STAGE must be teachers, scvi_condition, reuse, stack, select, selector_scores or finalize}" in
  teachers)
    mkdir -p "${ROOT}/methods"
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" --transductive "${common[@]}" --output "${ROOT}/methods/${method}"
    done
    "${PY}" scripts/run_leakage_safe_method.py --method graph_smooth --transductive --condition-column target \
      "${common[@]}" --output "${ROOT}/methods/graph_smooth_condition"
    ;;
  scvi_condition)
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${ROOT}/hybrid.h5ad" --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --output "${ROOT}/methods/scvi_standard_condition" --epochs 200 --seed 1729 --condition-column target
    "${PY}" scripts/count_scale_contract.py --contract "${ROOT}/methods/scvi_standard_condition" \
      --corrupted "${ROOT}/hybrid.h5ad" --output "${ROOT}/methods/scvi_inductive_condition"
    ;;
  reuse)
    "${PY}" scripts/count_scale_contract.py --contract "$(standard_dir magic)" --corrupted "${ROOT}/hybrid.h5ad" \
      --output "${ROOT}/methods/magic_inductive"
    "${PY}" scripts/count_scale_contract.py --contract "$(standard_dir scvi)" --corrupted "${ROOT}/hybrid.h5ad" \
      --output "${ROOT}/methods/scvi_inductive"
    ;;
  stack)
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${ROOT}/methods/safe_fusion" \
      --teacher-contract "${ROOT}/methods/gene_median" --teacher-contract "${ROOT}/methods/svd_impute" \
      --teacher-contract "${ROOT}/methods/graph_smooth" --teacher-contract "${ROOT}/methods/magic_inductive" \
      --teacher-contract "${ROOT}/methods/scvi_inductive"
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${ROOT}/methods/safe_fusion_condition" \
      --teacher-contract "${ROOT}/methods/gene_median" --teacher-contract "${ROOT}/methods/svd_impute" \
      --teacher-contract "${ROOT}/methods/graph_smooth_condition" --teacher-contract "${ROOT}/methods/magic_inductive" \
      --teacher-contract "${ROOT}/methods/scvi_inductive_condition"
    ;;
  select)
    mapfile -t genes < <(target_genes)
    plain=(--teacher-contract "${ROOT}/methods/gene_median" --teacher-contract "${ROOT}/methods/svd_impute" \
      --teacher-contract "${ROOT}/methods/graph_smooth" --teacher-contract "${ROOT}/methods/magic_inductive" \
      --teacher-contract "${ROOT}/methods/scvi_inductive")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/hybrid.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/methods/safe_fusion" "${plain[@]}" \
      --output-dir "${ROOT}/selector" --fit-split development --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --curve-points 2 --seed 1729 --score-gene "${genes[@]}"
    labelled=(--teacher-contract "${ROOT}/methods/gene_median" --teacher-contract "${ROOT}/methods/svd_impute" \
      --teacher-contract "${ROOT}/methods/graph_smooth_condition" --teacher-contract "${ROOT}/methods/magic_inductive" \
      --teacher-contract "${ROOT}/methods/scvi_inductive_condition")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/hybrid.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/methods/safe_fusion_condition" "${labelled[@]}" \
      --output-dir "${ROOT}/selector_condition" --fit-split development --condition-column target --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10 \
      --curve-points 2 --seed 1729 --score-gene "${genes[@]}"
    ;;
  selector_scores)

    [[ "${screen}" == norman_crispra ]] && exit 0
    mapfile -t genes < <(target_genes)
    OUT_ROOT="${OUT1}/review_root/selector_scores"
    plain=(--teacher-contract "${ROOT}/methods/gene_median" --teacher-contract "${ROOT}/methods/svd_impute" \
      --teacher-contract "${ROOT}/methods/graph_smooth" --teacher-contract "${ROOT}/methods/magic_inductive" \
      --teacher-contract "${ROOT}/methods/scvi_inductive")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/hybrid.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/methods/safe_fusion" "${plain[@]}" \
      --output-dir "${OUT_ROOT}/${screen}/safe_fusion" --fit-split development --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.1 --curve-points 2 --seed 1729 --score-gene "${genes[@]}"
    labelled=(--teacher-contract "${ROOT}/methods/gene_median" --teacher-contract "${ROOT}/methods/svd_impute" \
      --teacher-contract "${ROOT}/methods/graph_smooth_condition" --teacher-contract "${ROOT}/methods/magic_inductive" \
      --teacher-contract "${ROOT}/methods/scvi_inductive_condition")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${ROOT}/hybrid.h5ad" --truth "${truth}" \
      --coordinates "${ROOT}/coordinates.parquet" --splits "${ROOT}/splits.parquet" \
      --fusion-contract "${ROOT}/methods/safe_fusion_condition" "${labelled[@]}" \
      --output-dir "${OUT_ROOT}/${screen}/safe_fusion_condition" --fit-split development --condition-column target \
      --architecture mlp --budget-mode apply_topk --budgets 0.01 0.1 --curve-points 2 --seed 1729 --score-gene "${genes[@]}"
    ;;
  finalize)
    sources=()
    for pct in 1 2 3 4 5 6 7 8 9 10; do
      if (( pct == 10 )); then suffix=0p1; else suffix="0p0${pct}"; fi
      sources+=(--source "safe_fusion_${pct}pct=${ROOT}/selector/safe_fusion_calibrated_mlp_topk_${suffix}")
    done
    "${PY}" scripts/finalize_deployment_contracts.py --recorded "${ROOT}/recorded.h5ad" \
      --splits "${ROOT}/splits.parquet" "${sources[@]}" --output-root "${ROOT}"
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
