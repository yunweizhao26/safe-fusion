#!/usr/bin/env bash
#SBATCH --job-name=sf-baseline-env
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH --output=logs/slurm-baseline-env-%A_%a.out
#SBATCH --error=logs/slurm-baseline-env-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export CONDA_PKGS_DIRS="/tmp/safefusion-conda-${SLURM_JOB_ID}"

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  prefix=.conda-scvi-current
  specification=workflow/envs/scvi.yaml
else
  prefix=.conda-magic-current
  specification=workflow/envs/magic.yaml
fi

if [[ -x "${prefix}/bin/python" ]]; then
  conda env update \
    --prefix "${prefix}" \
    --file "${specification}"
else
  conda env create \
    --prefix "${prefix}" \
    --file "${specification}"
fi

"${prefix}/bin/python" --version
