#!/usr/bin/env bash
#SBATCH --job-name=sf-magic-current
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-magic-current-%A_%a.out
#SBATCH --error=logs/slurm-magic-current-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    corrupted=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/corrupted/mask_010.h5ad
    coordinates=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/coordinates/mask_010.parquet
    splits=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/splits.parquet
    output=artifacts/paper_evidence/baselines/magic/pancreas
    ;;
  1)
    corrupted=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/corrupted/mask_010.h5ad
    coordinates=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/coordinates/mask_010.parquet
    splits=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/splits.parquet
    output=artifacts/paper_evidence/baselines/magic/colon
    ;;
  2)
    corrupted=artifacts/paper_evidence/norman_crispra/corrupted.h5ad
    coordinates=artifacts/paper_evidence/norman_crispra/coordinates.parquet
    splits=artifacts/paper_evidence/norman_crispra/splits.parquet
    output=artifacts/paper_evidence/baselines/magic/norman
    ;;
esac

.conda-magic-current/bin/python scripts/run_magic_baseline.py \
  --corrupted "${corrupted}" \
  --coordinates "${coordinates}" \
  --splits "${splits}" \
  --output "${output}" \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --seed 1729
