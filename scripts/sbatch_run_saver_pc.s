#!/bin/bash
#SBATCH --job-name=sf_saver_pc
#SBATCH --output=logs/sf_saver_pc_%j.log
#SBATCH --error=logs/sf_saver_pc_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
PY="uv run python"

$PY -u scripts/run_saver_baseline.py \
  --corrupted artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/corrupted/mask_010.h5ad \
  --splits artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/splits.parquet \
  --output artifacts/paper_evidence/baselines/saver/pancreas_mask_010

$PY -u scripts/run_saver_baseline.py \
  --corrupted artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/corrupted/mask_010.h5ad \
  --splits artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/splits.parquet \
  --output artifacts/paper_evidence/baselines/saver/colon_mask_010

echo SAVER_PC_DONE
