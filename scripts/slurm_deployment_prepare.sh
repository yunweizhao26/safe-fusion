#!/usr/bin/env bash
#SBATCH --job-name=sf-deploy-prep
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-7%5
#SBATCH --output=logs/slurm-deployment-prepare-%A_%a.out
#SBATCH --error=logs/slurm-deployment-prepare-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

deployment_paths "${DEPLOY_KEYS[SLURM_ARRAY_TASK_ID]}"
step() { echo "step=$1 elapsed_s=${SECONDS}"; }

.venv/bin/python scripts/build_deployment_inputs.py --truth "${truth}" --corrupted "${input}" \
  --coordinates "${coordinates}" --splits "${splits}" --output-dir "${out}"
step inputs

for method in gene_median svd_impute graph_smooth; do
  .venv/bin/python scripts/run_leakage_safe_method.py --method "${method}" \
    --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
    --splits "${out}/splits.parquet" --output "${deploy_methods}/${method}" --seed 1729
  step "${method}"
done

.conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
  --input "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" \
  --splits "${out}/splits.parquet" --output "${deploy_methods}/magic_inductive" \
  --seed 1729 --n-jobs "${SLURM_CPUS_PER_TASK}"
step magic_inductive
