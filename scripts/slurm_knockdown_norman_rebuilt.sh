#!/usr/bin/env bash
#SBATCH --job-name=sf-kd-norman
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-kd-norman-%j.out
#SBATCH --error=logs/slurm-kd-norman-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh
source scripts/knockdown_paths.sh
knockdown_paths norman_crispra

PY=.venv/bin/python
FRACTIONS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
step() { echo "step=$1 elapsed_s=${SECONDS}"; }
suffix_for_pct() {
  if (( $1 == 10 )); then echo 0p1; else echo "0p0$1"; fi
}
condition_teachers() {
  local plain="$1" labelled="$2"
  printf -- '--teacher-contract\n%s\n' "${plain}/gene_median" "${plain}/svd_impute" \
    "${labelled}/graph_smooth_condition" "${plain}/magic_inductive" "${labelled}/scvi_inductive_condition"
}
target_genes() {
  "${PY}" - "$1" <<'EOF'
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

case "${STAGE:?STAGE must be masked_knn, masked_scvi, masked_select, deploy_prepare, deploy_scvi or deploy_fill}" in
  masked_knn)
    "${PY}" scripts/run_leakage_safe_method.py --method graph_smooth --condition-column target \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${label_root}/graph_smooth_condition" --seed 1729
    ;;
  masked_scvi)
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi --condition-column target \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${label_root}/scvi_inductive_condition" --seed 1729
    ;;
  masked_select)
    mapfile -t teacher_args < <(condition_teachers "${methods_root}" "${label_root}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${label_root}/safe_fusion_condition" --seed 1729 "${teacher_args[@]}"
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${input}" --truth "${truth}" --coordinates "${coordinates}" --splits "${splits}" \
      --fusion-contract "${label_root}/safe_fusion_condition" "${teacher_args[@]}" \
      --output-dir "${label_root}/selector_condition" --fit-split development --condition-column target \
      --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729
    ;;
  deploy_prepare)
    "${PY}" scripts/build_deployment_inputs.py --truth "${truth}" --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output-dir "${out}"
    step inputs
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
        --splits "${out}/splits.parquet" --output "${deploy_methods}/${method}" --seed 1729
      step "${method}"
    done
    "${PY}" scripts/run_leakage_safe_method.py --method graph_smooth --condition-column target \
      --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
      --splits "${out}/splits.parquet" --output "${deploy_methods}/graph_smooth_condition" --seed 1729
    step graph_smooth_condition
    .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
      --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
      --splits "${out}/splits.parquet" --output "${deploy_methods}/magic_inductive" \
      --seed 1729 --n-jobs "${SLURM_CPUS_PER_TASK}"
    step magic_inductive
    ;;
  deploy_scvi)
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
      --splits "${out}/splits.parquet" --output "${deploy_methods}/scvi_inductive" --seed 1729
    step scvi_inductive
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi --condition-column target \
      --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
      --splits "${out}/splits.parquet" --output "${deploy_methods}/scvi_inductive_condition" --seed 1729
    step scvi_inductive_condition
    ;;
  deploy_fill)
    common=(--coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet")
    mapfile -t genes < <(target_genes "${out}/hybrid.h5ad")
    mapfile -t teacher_args < <(teacher_contract_args "${deploy_methods}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --input "${out}/hybrid.h5ad" "${common[@]}" \
      --output "${deploy_methods}/safe_fusion" --seed 1729 "${teacher_args[@]}"
    step safe_fusion
    "${PY}" scripts/calibrated_selective_fill.py --corrupted "${out}/hybrid.h5ad" --truth "${truth}" "${common[@]}" \
      --fusion-contract "${deploy_methods}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${out}/selector" "${fit[@]}" \
      --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
      --curve-points 2 --seed 1729 --score-gene "${genes[@]}"
    step selector
    for spec in "svd_impute svd" "graph_smooth weighted_knn"; do
      set -- ${spec}
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${out}/hybrid.h5ad" --splits "${out}/splits.parquet" \
        --method-contract "${deploy_methods}/$1" --method-name "$2" \
        --output-root "${out}/matched" --fractions "${FRACTIONS[@]}"
    done
    step matched
    sources=(
      --source "gene_median=${deploy_methods}/gene_median"
      --source "svd_impute=${deploy_methods}/svd_impute"
      --source "graph_smooth=${deploy_methods}/graph_smooth"
      --source "safe_fusion=${deploy_methods}/safe_fusion"
    )
    for pct in 1 2 3 4 5 6 7 8 9 10; do
      suffix="$(suffix_for_pct "${pct}")"
      sources+=(--source "safe_fusion_${pct}pct=${out}/selector/safe_fusion_calibrated_mlp_topk_${suffix}")
      sources+=(--source "svd_${pct}pct=${out}/matched/svd_topk_${suffix}")
      sources+=(--source "weighted_knn_${pct}pct=${out}/matched/weighted_knn_topk_${suffix}")
    done
    "${PY}" scripts/finalize_deployment_contracts.py --recorded "${out}/recorded.h5ad" \
      --splits "${out}/splits.parquet" "${sources[@]}" --output-root "${out}"
    step finalize
    mapfile -t condition_args < <(condition_teachers "${deploy_methods}" "${deploy_methods}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --input "${out}/hybrid.h5ad" "${common[@]}" \
      --output "${deploy_methods}/safe_fusion_condition" --seed 1729 "${condition_args[@]}"
    "${PY}" scripts/calibrated_selective_fill.py --corrupted "${out}/hybrid.h5ad" --truth "${truth}" "${common[@]}" \
      --fusion-contract "${deploy_methods}/safe_fusion_condition" "${condition_args[@]}" \
      --output-dir "${out}/selector_condition" --fit-split development --condition-column target \
      --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
      --curve-points 2 --seed 1729 --score-gene "${genes[@]}"
    step selector_condition
    for spec in "graph_smooth_condition knn_condition" "scvi_inductive_condition scvi_condition" \
                "scvi_inductive scvi" "magic_inductive magic"; do
      set -- ${spec}
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${out}/hybrid.h5ad" --splits "${out}/splits.parquet" \
        --method-contract "${deploy_methods}/$1" --method-name "$2" \
        --output-root "${out}/matched_condition" --fractions "${FRACTIONS[@]}"
    done
    step matched_condition
    ;;
esac
