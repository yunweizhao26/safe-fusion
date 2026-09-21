#!/bin/bash
#SBATCH --job-name=sf_gears
#SBATCH --output=logs/sf_gears_%j.log
#SBATCH --error=logs/sf_gears_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Official GEARS perturbation-response sweep over imputed inputs.
# This must run on a Slurm node: the login node caps processes at ~1-1.5 GB,
# while GEARS training needs 8-16 GB with the full GO/co-expression graphs.

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

.venv-gears/bin/python -u scripts/perturbation_response_gears.py \
  --truth external_data/prepared/norman_crispra.h5ad \
  --corrupted artifacts/paper_evidence/norman_crispra/corrupted.h5ad \
  --splits artifacts/paper_evidence/norman_crispra/splits.parquet \
  --method graph_smooth=artifacts/paper_evidence/norman_crispra/methods/graph_smooth \
  --method safe_fusion=artifacts/paper_evidence/norman_crispra/methods/safe_fusion \
  --method calibrated_0p02=artifacts/paper_evidence/norman_calibrated/safe_fusion_calibrated_0p02 \
  --method calibrated_0p05=artifacts/paper_evidence/norman_calibrated/safe_fusion_calibrated_0p05 \
  --output-dir artifacts/paper_evidence/norman_perturbation_response_gears_final \
  --epochs 20 --hidden-size 64 --seed 1 \
  --cells-per-condition 30 --control-cells 600 \
  --gene-cap 1500 --go-top-k 20 --batch-size 16

echo GEARS_SWEEP_DONE
