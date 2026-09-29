#!/usr/bin/env bash
#SBATCH --job-name=sf-dc-analysis
#SBATCH --account=torch_pr_634_general
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-9
#SBATCH --output=logs/slurm-disease-control-analysis-%A_%a.out
#SBATCH --error=logs/slurm-disease-control-analysis-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

ANALYSES=(condition effects modules annotation sex_zeros)
TISSUES=(pancreas colon)
analysis="${ANALYSES[SLURM_ARRAY_TASK_ID / 2]}"
tissue="${TISSUES[SLURM_ARRAY_TASK_ID % 2]}"

PY=.venv/bin/python
if [[ "${analysis}" == modules || "${analysis}" == annotation ]]; then
  PY=.venv-scanpy/bin/python
  if [[ ! -x "${PY}" ]]; then
    uv venv --python 3.12 .venv-scanpy
    uv pip install --python "${PY}" numpy==1.26.4 scipy==1.15.2 pandas==2.2.3 anndata==0.11.4 \
      scikit-learn==1.7.2 pyarrow==19.0.1 h5py==3.12.1 scanpy==1.11.5 leidenalg==0.12.0 igraph==1.0.0
  fi
fi

"${PY}" "scripts/disease_control_${analysis}.py" --tissue "${tissue}"
echo "analysis=${analysis} tissue=${tissue} elapsed_s=${SECONDS}"
