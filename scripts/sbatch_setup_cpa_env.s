#!/bin/bash
#SBATCH --job-name=sf_cpa_env
#SBATCH --output=logs/sf_cpa_env_%j.log
#SBATCH --error=logs/sf_cpa_env_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# Build the CPA environment on a compute node (login node memory cap is too
# small for this dependency set). Must run before the CPA sweep job.

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export PIP_CACHE_DIR="${TMPDIR:-/tmp}/safefusion-pip"
mkdir -p "$PIP_CACHE_DIR"
: "${CPA_ROOT:?set CPA_ROOT to the official CPA checkout}"

if [ ! -x .venv-cpa/bin/python ]; then
  uv run python -m venv .venv-cpa
fi
.venv-cpa/bin/pip install -q -U pip
.venv-cpa/bin/pip install -q "torch==2.0.1" --index-url https://download.pytorch.org/whl/cpu
.venv-cpa/bin/pip install -q -e "$CPA_ROOT"
# scvi-tools/ray need pyarrow with PyExtensionType; pin a compatible version.
.venv-cpa/bin/pip install -q "pyarrow==14.0.2"

echo CPA_ENV_OK
.venv-cpa/bin/python -c "import cpa; print('cpa import ok', cpa.__file__)"
