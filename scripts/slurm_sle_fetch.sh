#!/usr/bin/env bash
#SBATCH --job-name=sle-fetch
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs,cpu_short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/slurm-sle-fetch-%j.out
#SBATCH --error=logs/slurm-sle-fetch-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
VERSION="${VERSION:-c55dc602-d168-4d15-acc1-5de4f2f5d551}"
TARGET="external_data/cellxgene/${VERSION}.h5ad"
OUT=artifacts/paper_evidence/sle_treg_case
mkdir -p external_data/cellxgene "${OUT}"

if [[ ! -s "${TARGET}" ]]; then
  curl -fsSL --retry 5 -C - -o "${TARGET}.part" "https://datasets.cellxgene.cziscience.com/${VERSION}.h5ad"
  mv "${TARGET}.part" "${TARGET}"
fi
curl -fsS "https://api.cellxgene.cziscience.com/curation/v1/collections/436154da-bcf1-4130-9c8b-120ff9a888f2" \
  > "${OUT}/cellxgene_collection.json"
digest="$(sha256sum "${TARGET}" | cut -d' ' -f1)"
printf '{"file": "%s", "dataset_version_id": "%s", "sha256": "%s", "bytes": %s}\n' \
  "${TARGET}" "${VERSION}" "${digest}" "$(stat -c %s "${TARGET}")" > "${OUT}/source_file.json"
cat "${OUT}/source_file.json"
if [[ -n "${EXPECTED_SHA256:-}" && "${digest}" != "${EXPECTED_SHA256}" ]]; then
  echo "SHA-256 mismatch: ${digest}" >&2
  exit 1
fi
