#!/usr/bin/env bash
#SBATCH --job-name=sf-papalexi-audit
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-papalexi-audit-%j.out
#SBATCH --error=logs/slurm-papalexi-audit-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.conda-scvi-current/bin/python scripts/audit_papalexi_multimodal.py \
  --mudata external_data/papalexi_multimodal/papalexi.h5mu \
  --rna external_data/scperturb/PapalexiSatija2021_eccite_RNA.h5ad \
  --geo-raw-tar external_data/papalexi_multimodal/GSE153056_RAW.tar \
  --output-dir artifacts/paper_evidence/papalexi_crossmodal/audit
