#!/usr/bin/env bash
#SBATCH --job-name=sf-dc-fill
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-3
#SBATCH --output=logs/slurm-disease-control-fill-%A_%a.out
#SBATCH --error=logs/slurm-disease-control-fill-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

PY=.venv/bin/python
DC_ROOT="${DC_ROOT:-artifacts/paper_evidence/disease_control_checks}"
FRACTIONS=(0.01 0.05 0.10)
SEX_GENES=(ENSG00000229807 ENSG00000129824)
KEYS=(pancreas_0 pancreas_1 pancreas_2 colon)
key="${KEYS[SLURM_ARRAY_TASK_ID]}"
step() { echo "step=$1 elapsed_s=${SECONDS}"; }

deployment_paths "${key}"
for spec in "magic_inductive magic" "scvi_inductive scvi"; do
  set -- ${spec}
  "${PY}" scripts/apply_fill_fraction.py --corrupted "${out}/hybrid.h5ad" --splits "${out}/splits.parquet" \
    --method-contract "${deploy_methods}/$1" --method-name "$2" \
    --output-root "${DC_ROOT}/deployment_fills/${key}" --fractions "${FRACTIONS[@]}"
done
step matched_fills

mapfile -t teacher_args < <(teacher_contract_args "${deploy_methods}")
"${PY}" scripts/calibrated_selective_fill.py \
  --corrupted "${out}/hybrid.h5ad" --truth "${truth}" \
  --coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet" \
  --fusion-contract "${deploy_methods}/safe_fusion" "${teacher_args[@]}" \
  --output-dir "${DC_ROOT}/selector_scores/deployment/${key}" "${fit[@]}" \
  --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
  --curve-points 2 --score-gene "${SEX_GENES[@]}" --seed 1729
step deployment_selector

unit_paths "${key}"
mapfile -t teacher_args < <(teacher_contract_args "${methods_root}")
"${PY}" scripts/calibrated_selective_fill.py \
  --corrupted "${input}" --truth "${truth}" \
  --coordinates "${coordinates}" --splits "${splits}" \
  --fusion-contract "${methods_root}/safe_fusion" "${teacher_args[@]}" \
  --output-dir "${DC_ROOT}/selector_scores/masked/${key}" "${fit[@]}" \
  --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
  --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
  --score-gene "${SEX_GENES[@]}" --seed 1729
step masked_selector
