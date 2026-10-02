#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}
export NUMBA_CACHE_DIR=$PWD/artifacts/paper_evidence/review_round4/round2_extras/cache
export MPLCONFIGDIR=$PWD/artifacts/paper_evidence/review_round4/round2_extras/mpl
exec .venv/bin/python -u "$@"
