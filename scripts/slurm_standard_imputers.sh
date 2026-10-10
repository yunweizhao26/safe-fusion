#!/usr/bin/env bash
#SBATCH --job-name=sf-standard-imputers
#SBATCH --account=torch_pr_634_general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=logs/slurm-standard-imputers-%A_%a.out
#SBATCH --error=logs/slurm-standard-imputers-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUDA_VISIBLE_DEVICES=""

DATASET=63ff2c52-cb63-44f0-bac3-d0b33373e312
SOURCE="${SOURCE:-external_data/cellxgene/${DATASET}.h5ad}"
SOURCE_URL="https://datasets.cellxgene.cziscience.com/${DATASET}.h5ad"
SOURCE_SHA256=612e03925bd5860e1ec81b0295af40a7f9e4111638f8f41ccb2fe976f428f053
ENV_PREFIX="${ENV_PREFIX:-.conda-standard-imputers}"
OUT="${OUT:-artifacts/paper_evidence/standard_imputers}"
SEED="${SEED:-1729}"
SUBSETS=(
  "Crohn disease|caecum"
  "Crohn disease|caecum epithelium"
  "Crohn disease|right colon"
  "Crohn disease|lamina propria of mucosa of colon"
)
methods=(SAUCIE MAGIC deepImpute scScope scVI knn_smoothing)
method="${methods[SLURM_ARRAY_TASK_ID]}"

mkdir -p "$(dirname "${SOURCE}")" logs
(
  flock 9
  if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    CONDA_PKGS_DIRS="/tmp/safefusion-conda-${SLURM_JOB_ID}" \
      conda env create --prefix "${ENV_PREFIX}" --file scripts/standard_imputers/environment.yaml
  fi
  if [[ ! -f "${SOURCE}" ]]; then
    curl -L --fail -o "${SOURCE}.part" "${SOURCE_URL}"
    mv "${SOURCE}.part" "${SOURCE}"
  fi
) 9>logs/standard-imputers-setup.lock
echo "${SOURCE_SHA256}  ${SOURCE}" | sha256sum --check --quiet

subset_args=()
for subset in "${SUBSETS[@]}"; do
  subset_args+=(--subset "${subset}")
done
"${ENV_PREFIX}/bin/python" scripts/standard_imputers/run_standard_imputers.py \
  --method "${method}" --input "${SOURCE}" --output-root "${OUT}" --seed "${SEED}" "${subset_args[@]}"
