#!/usr/bin/env bash
#SBATCH --job-name=sle-env
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs,cpu_short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-sle-env-%j.out
#SBATCH --error=logs/slurm-sle-env-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
UV="${UV:-uv}"
"${UV}" venv --python 3.12 .venv-sle
"${UV}" pip install --python .venv-sle/bin/python -r configs/sle_requirements.txt
.venv-sle/bin/python -c "import scanpy, leidenalg; print(scanpy.__version__, leidenalg.__version__)"
