#!/bin/bash
#SBATCH --job-name=sf_base_env
#SBATCH --output=logs/sf_base_env_%j.log
#SBATCH --error=logs/sf_base_env_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# Build the Python official-baseline environments (scGPT, scGCL) and download
# the scGPT human checkpoint. Runs on a compute node so the login node memory
# cap is not an issue.

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

export PIP_CACHE_DIR="${TMPDIR:-/tmp}/safefusion-pip"
mkdir -p "$PIP_CACHE_DIR" external_data/baselines

if [ ! -x .venv-baselines/bin/python ]; then
  uv run python -m venv .venv-baselines
fi
.venv-baselines/bin/pip install -q -U pip
.venv-baselines/bin/pip install -q "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
.venv-baselines/bin/pip install -q scgpt
.venv-baselines/bin/pip install -q "gdown"

# scGCL best-effort via GitHub archive (the git+ pip install needs credentials
# from this node; if the archive is unavailable, scGPT proceeds alone).
set +e
cd external_data/baselines
curl -L --fail --max-time 120 https://codeload.github.com/PharrellWANG/scGCL/zip/refs/heads/main -o scgcl.zip
CURL_OK=$?
set -e
if [ "$CURL_OK" -eq 0 ] && [ -f scgcl.zip ]; then
  unzip -q scgcl.zip
  cd "$SAFE_FUSION_ROOT"
  .venv-baselines/bin/pip install -q -e external_data/baselines/scGCL-main
  echo SCGCL_OK
else
  echo SCGCL_SKIPPED
fi
cd "$SAFE_FUSION_ROOT"

mkdir -p external_data/baselines/scgpt_human
.venv-baselines/bin/gdown --folder \
  "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" \
  -O external_data/baselines/scgpt_human

echo PYTHON_BASELINES_OK
.venv-baselines/bin/python -c "import scgpt; print('scgpt ok')"
