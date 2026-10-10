#!/bin/bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
O="$PWD/artifacts/paper_evidence/review_round4/fill_grid_dense"
export MPLCONFIGDIR="$O/.mpl" NUMBA_CACHE_DIR="$O/.numba" XDG_CACHE_HOME="$O/.cache" TMPDIR="$O/tmp"
C=scripts/analyses/fill_grid_dense
case "$1" in
  gate-unit) .venv/bin/python "$C/unit.py" gate "$SLURM_ARRAY_TASK_ID" ;;
  gate) .venv/bin/python "$C/run.py" gate ;;
  dense-unit) .venv/bin/python "$C/unit.py" dense "$SLURM_ARRAY_TASK_ID" ;;
  summary)
    .venv/bin/python "$C/summarize.py"
    .venv/bin/python "$C/verify.py"
    .venv/bin/python scripts/plot_fill_grid_main.py
    ;;
  *) exit 2 ;;
esac
