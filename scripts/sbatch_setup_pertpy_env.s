#!/usr/bin/env bash
#SBATCH --job-name=sf-setup-pertpy
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-setup-pertpy-%j.out
#SBATCH --error=logs/slurm-setup-pertpy-%j.err

set -euo pipefail
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
ENV_PREFIX="${ENV_PREFIX:-.venv-pertpy}"
UV="${UV:-uv}"
"${UV}" venv --python 3.12 "${ENV_PREFIX}"
"${UV}" pip install --python "${ENV_PREFIX}/bin/python" "pertpy==1.3.0" "mudata==0.4.1" "scanpy==1.12.4"
"${UV}" pip freeze --python "${ENV_PREFIX}/bin/python" > "${ENV_PREFIX}/requirements.lock"
"${ENV_PREFIX}/bin/python" -c "from importlib.metadata import version; print({name: version(name) for name in ('pertpy', 'scanpy', 'mudata')})"
