#!/usr/bin/env bash
#SBATCH --job-name=sf-pap-xm-prep
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-pap-xm-prep-%j.out
#SBATCH --error=logs/slurm-pap-xm-prep-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
PY=.conda-scvi-current/bin/python

"${PY}" scripts/prepare_external_perturbseq.py \
  --input external_data/scperturb/PapalexiSatija2021_eccite_RNA.h5ad \
  --output external_data/prepared/papalexi_eccite_crossmodal.h5ad \
  --dataset papalexi_eccite \
  --include-gene CD274 CD86 PDCD1LG2 HAVCR2 \
  --seed 1729

"${PY}" scripts/build_papalexi_crossmodal_benchmark.py \
  --input external_data/prepared/papalexi_eccite_crossmodal.h5ad \
  --output-dir artifacts/paper_evidence/papalexi_crossmodal/benchmark \
  --seed 1729
