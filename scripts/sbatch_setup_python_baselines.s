#!/bin/bash
#SBATCH --job-name=sf_base_env
#SBATCH --output=logs/sf_base_env_%j.log
#SBATCH --error=logs/sf_base_env_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

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

.venv-baselines/bin/pip install -q "faiss-cpu==1.9.0.post1"
if [ ! -d baselines_and_data/scGCL/.git ]; then
  git clone https://github.com/zehaoxiong123/scGCL.git baselines_and_data/scGCL
fi
git -C baselines_and_data/scGCL checkout -q 317015acdf06d2929c20a7d2858bac539b3d8ebd

mkdir -p external_data/baselines/scgpt_human
.venv-baselines/bin/gdown --folder \
  "https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y" \
  -O external_data/baselines/scgpt_human

echo PYTHON_BASELINES_OK
.venv-baselines/bin/python -c "import scgpt; print('scgpt ok')"
