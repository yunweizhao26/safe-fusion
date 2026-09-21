#!/usr/bin/env bash
#SBATCH --job-name=sf-sam-panel
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-sam-panel-%j.out
#SBATCH --error=logs/slurm-sam-panel-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
PY=.conda-scvi-current/bin/python
OUT=artifacts/paper_evidence/papalexi_crossmodal/samuel_reproduction
mkdir -p "${OUT}"

"${PY}" safe-fusion-stn/build_real_panel.py \
  --mudata external_data/papalexi_multimodal/papalexi.h5mu \
  --rna external_data/scperturb/PapalexiSatija2021_eccite_RNA.h5ad \
  --out "${OUT}/eccite_validation_panel_real.csv"

"${PY}" safe-fusion-stn/make_test3_panel_clr.py \
  --panel "${OUT}/eccite_validation_panel_real.csv" \
  --out "${OUT}/eccite_panel_pdl1_clr.csv" \
  --protein PDL1 \
  --gene CD274

"${PY}" scripts/compare_samuel_papalexi_panel.py \
  --samuel-panel "${OUT}/eccite_validation_panel_real.csv" \
  --audit-panel artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet \
  --output "${OUT}/comparison.json"
