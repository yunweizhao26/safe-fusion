#!/bin/bash
#SBATCH --job-name=sf_cpa
#SBATCH --output=logs/sf_cpa_%j.log
#SBATCH --error=logs/sf_cpa_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# CPA perturbation-response sweep over imputed inputs (same held-out
# conditions as the GEARS sweep). Requires the CPA env built by
# scripts/sbatch_setup_cpa_env.s.

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

.venv-cpa/bin/python -u scripts/perturbation_response_cpa.py \
  --truth external_data/prepared/norman_crispra.h5ad \
  --corrupted artifacts/paper_evidence/norman_crispra/corrupted.h5ad \
  --splits artifacts/paper_evidence/norman_crispra/splits.parquet \
  --method graph_smooth=artifacts/paper_evidence/norman_crispra/methods/graph_smooth \
  --method safe_fusion=artifacts/paper_evidence/norman_crispra/methods/safe_fusion \
  --method calibrated_0p02=artifacts/paper_evidence/norman_calibrated/safe_fusion_calibrated_0p02 \
  --method calibrated_0p05=artifacts/paper_evidence/norman_calibrated/safe_fusion_calibrated_0p05 \
  --output-dir artifacts/paper_evidence/norman_perturbation_response_cpa \
  --cells-per-condition 30 --control-cells 600 --gene-cap 1500 \
  --max-epochs 300 --seed 8206 \
  --test-conditions AHR BAK1 C19orf26 CBL CDKN1A CDKN1C CEBPA CITED1 CLDN6 COL2A1 CSRNP1 HK2 KLF1 MAPK1 NCL TSC22D1

echo CPA_SWEEP_DONE
