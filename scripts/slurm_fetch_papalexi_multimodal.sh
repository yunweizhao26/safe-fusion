#!/usr/bin/env bash
#SBATCH --job-name=sf-papalexi-fetch
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/slurm-papalexi-fetch-%j.out
#SBATCH --error=logs/slurm-papalexi-fetch-%j.err

set -euo pipefail

ROOT=$(pwd)
OUT=${ROOT}/external_data/papalexi_multimodal
mkdir -p "${OUT}"

curl --location --fail --retry 5 --continue-at - \
  --output "${OUT}/papalexi.h5mu" \
  https://ndownloader.figshare.com/files/36509460
printf '%s  %s\n' \
  419ec1f14615b4423edd6f16b65c55a7 \
  "${OUT}/papalexi.h5mu" | md5sum --check --status

curl --location --fail --retry 5 --continue-at - \
  --output "${OUT}/GSE153056_RAW.tar" \
  'https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE153056&format=file'

md5sum "${OUT}/papalexi.h5mu" > "${OUT}/checksums.md5"
sha256sum "${OUT}/GSE153056_RAW.tar" > "${OUT}/checksums.sha256"
tar -tf "${OUT}/GSE153056_RAW.tar" > "${OUT}/GSE153056_RAW.contents.txt"
