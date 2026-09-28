#!/usr/bin/env bash
#SBATCH --job-name=sf-deploy-fill
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-8%5
#SBATCH --output=logs/slurm-deployment-fill-%A_%a.out
#SBATCH --error=logs/slurm-deployment-fill-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

PY=.venv/bin/python
PCTS="${PCTS:-1 2 3 4 5 6 7 8 9 10}"
deployment_paths "${DEPLOY_KEYS[SLURM_ARRAY_TASK_ID]}"
mapfile -t teacher_args < <(teacher_contract_args "${deploy_methods}")
step() { echo "step=$1 elapsed_s=${SECONDS}"; }
suffix_for_pct() {
  if (( $1 == 10 )); then echo 0p1; else echo "0p0$1"; fi
}

"${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
  --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
  --splits "${out}/splits.parquet" --output "${deploy_methods}/safe_fusion" \
  --seed 1729 "${teacher_args[@]}"
step safe_fusion

fractions=()
for pct in ${PCTS}; do fractions+=("$(printf '0.%02d' "${pct}")"); done
"${PY}" scripts/calibrated_selective_fill.py \
  --corrupted "${out}/hybrid.h5ad" --truth "${truth}" \
  --coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet" \
  --fusion-contract "${deploy_methods}/safe_fusion" "${teacher_args[@]}" \
  --output-dir "${out}/selector" "${fit[@]}" \
  --architecture mlp --budget-mode apply_topk --budgets "${fractions[@]}" \
  --curve-points 2 --seed 1729
step selector

for spec in "svd_impute svd" "graph_smooth weighted_knn"; do
  set -- ${spec}
  "${PY}" scripts/apply_fill_fraction.py --corrupted "${out}/hybrid.h5ad" --splits "${out}/splits.parquet" \
    --method-contract "${deploy_methods}/$1" --method-name "$2" \
    --output-root "${out}/matched" --fractions "${fractions[@]}"
done
step matched

sources=(
  --source "gene_median=${deploy_methods}/gene_median"
  --source "svd_impute=${deploy_methods}/svd_impute"
  --source "graph_smooth=${deploy_methods}/graph_smooth"
  --source "safe_fusion=${deploy_methods}/safe_fusion"
)
for pct in ${PCTS}; do
  suffix="$(suffix_for_pct "${pct}")"
  sources+=(--source "safe_fusion_${pct}pct=${out}/selector/safe_fusion_calibrated_mlp_topk_${suffix}")
  sources+=(--source "svd_${pct}pct=${out}/matched/svd_topk_${suffix}")
  sources+=(--source "weighted_knn_${pct}pct=${out}/matched/weighted_knn_topk_${suffix}")
done
"${PY}" scripts/finalize_deployment_contracts.py --recorded "${out}/recorded.h5ad" \
  --splits "${out}/splits.parquet" "${sources[@]}" --output-root "${out}"
step finalize
