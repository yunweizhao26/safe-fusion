#!/usr/bin/env bash
#SBATCH --job-name=sle-fetch-cite
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs,cpu_short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm-sle-fetch-cite-%j.out
#SBATCH --error=logs/slurm-sle-fetch-cite-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
DEST=external_data/citeseq_hao2021
mkdir -p "${DEST}"
GEO=https://ftp.ncbi.nlm.nih.gov/geo
FILES=(
  "samples/GSM5008nnn/GSM5008737/suppl/GSM5008737_RNA_3P-matrix.mtx.gz"
  "samples/GSM5008nnn/GSM5008737/suppl/GSM5008737_RNA_3P-barcodes.tsv.gz"
  "samples/GSM5008nnn/GSM5008737/suppl/GSM5008737_RNA_3P-features.tsv.gz"
  "samples/GSM5008nnn/GSM5008738/suppl/GSM5008738_ADT_3P-matrix.mtx.gz"
  "samples/GSM5008nnn/GSM5008738/suppl/GSM5008738_ADT_3P-barcodes.tsv.gz"
  "samples/GSM5008nnn/GSM5008738/suppl/GSM5008738_ADT_3P-features.tsv.gz"
  "series/GSE164nnn/GSE164378/suppl/GSE164378_sc.meta.data_3P.csv.gz"
)
for path in "${FILES[@]}"; do
  name="$(basename "${path}")"
  [[ -s "${DEST}/${name}" ]] || curl -fsSL --retry 5 -o "${DEST}/${name}" "${GEO}/${path}"
done
(cd "${DEST}" && sha256sum *.gz > SHA256SUMS)
cat "${DEST}/SHA256SUMS"
