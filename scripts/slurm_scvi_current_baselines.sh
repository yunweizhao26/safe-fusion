#!/usr/bin/env bash
#SBATCH --job-name=sf-scvi-current
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-1
#SBATCH --output=logs/slurm-scvi-current-%A_%a.out
#SBATCH --error=logs/slurm-scvi-current-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    corrupted=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/corrupted/mask_010.h5ad
    coordinates=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/coordinates/mask_010.parquet
    splits=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/splits.parquet
    output=artifacts/paper_evidence/baselines/scvi/pancreas
    ;;
  1)
    corrupted=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/corrupted/mask_010.h5ad
    coordinates=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/coordinates/mask_010.parquet
    splits=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/splits.parquet
    output=artifacts/paper_evidence/baselines/scvi/colon
    ;;
esac

.conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
  --corrupted "${corrupted}" \
  --coordinates "${coordinates}" \
  --splits "${splits}" \
  --output "${output}" \
  --epochs 200 \
  --seed 1729
