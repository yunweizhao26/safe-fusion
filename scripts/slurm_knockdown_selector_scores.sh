#!/usr/bin/env bash
#SBATCH --job-name=sf-kd-scores
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-kd-scores-%A_%a.out
#SBATCH --error=logs/slurm-kd-scores-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

screens=(adamson_crispri papalexi_eccite)
variants=(safe_fusion safe_fusion_condition)
screen="${screens[SLURM_ARRAY_TASK_ID / 2]}"
variant="${variants[SLURM_ARRAY_TASK_ID % 2]}"
deployment_paths "${screen}"
OUT_ROOT="${OUT_ROOT:-artifacts/paper_evidence/review_round2/knockdown/selector_scores}"
output="${OUT_ROOT}/${screen}/${variant}"

if [[ "${variant}" == safe_fusion ]]; then
  teachers=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)
  condition=()
else
  teachers=(gene_median svd_impute graph_smooth_condition magic_inductive scvi_inductive_condition)
  condition=(--condition-column target)
fi
teacher_args=()
for name in "${teachers[@]}"; do teacher_args+=(--teacher-contract "${deploy_methods}/${name}"); done

mapfile -t genes < <(.venv/bin/python - "${out}/hybrid.h5ad" <<'EOF'
import sys
import anndata as ad
adata = ad.read_h5ad(sys.argv[1], backed="r")
symbols = adata.var["feature_name"].astype(str) if "feature_name" in adata.var else adata.var_names.astype(str)
lookup = dict(zip(symbols, adata.var_names.astype(str)))
for target in sorted(set(adata.obs["target"].astype(str)) - {"none"}):
    if target in lookup:
        print(lookup[target])
EOF
)

.venv/bin/python scripts/calibrated_selective_fill.py \
  --corrupted "${out}/hybrid.h5ad" --truth "${truth}" \
  --coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet" \
  --fusion-contract "${deploy_methods}/${variant}" "${teacher_args[@]}" \
  --output-dir "${output}" "${fit[@]}" "${condition[@]}" \
  --architecture mlp --budget-mode apply_topk --budgets 0.01 0.1 \
  --curve-points 2 --seed 1729 --score-gene "${genes[@]}"
