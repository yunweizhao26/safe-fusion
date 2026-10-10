#!/bin/bash
#SBATCH --job-name=safefusion-scgcl
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/scgcl_%A_%a.out
#SBATCH --array=0-1%1

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export MPLCONFIGDIR="$PWD/.matplotlib-cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    INPUT=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/corrupted/mask_010.h5ad
    COORDINATES=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/coordinates/mask_010.parquet
    SPLITS=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/splits.parquet
    OUTPUT=artifacts/paper_evidence/baselines/scgcl/pancreas
    ;;
  1)
    INPUT=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/corrupted/mask_010.h5ad
    COORDINATES=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/coordinates/mask_010.parquet
    SPLITS=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/splits.parquet
    OUTPUT=artifacts/paper_evidence/baselines/scgcl/colon
    ;;
  *)
    echo "unexpected array index: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

RESUME_ARGS=()
if [[ -f "${OUTPUT}/checkpoint.pt" ]]; then
  RESUME_ARGS=(--resume)
fi

.venv-baselines/bin/python scripts/run_scgcl_baseline.py \
  --corrupted "${INPUT}" \
  --output "${OUTPUT}" \
  --epochs 300 \
  --lr 1e-6 \
  --stability-note "Post-default stability sensitivity: on pancreas lr=1e-3 became non-finite at epoch 68 under CUDA-compatible and CPU execution and lr=1e-4 at epoch 165; lr=1e-5 completed pancreas but became non-finite after epoch 240 on colon; all other pinned settings are unchanged." \
  --device cpu \
  --seed 0 \
  "${RESUME_ARGS[@]}"

.venv-baselines/bin/python scripts/evaluate_scgcl_baseline.py \
  --corrupted "${INPUT}" \
  --coordinates "${COORDINATES}" \
  --splits "${SPLITS}" \
  --method-output "${OUTPUT}" \
  --output "${OUTPUT}/evaluation.json" \
  --bootstrap-replicates 2000 \
  --seed 1729
