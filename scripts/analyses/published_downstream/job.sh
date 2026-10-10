#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${NUM_THREADS:-$SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
export TMPDIR="$PWD/artifacts/paper_evidence/review_round4/published_downstream/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
export NUMBA_CACHE_DIR="$TMPDIR/numba" MPLCONFIGDIR="$TMPDIR/mpl" XDG_CACHE_HOME="$TMPDIR/cache"
mkdir -p "$TMPDIR"
script=$1
shift
if [[ "$script" == */fit.py || "$script" == scripts/analyses/published_downstream/evaluate.py || "$script" == scripts/analyses/published_downstream_allcells/evaluate.py ]]; then
    exec .venv/bin/python -u "$script" "$@" "${SLURM_ARRAY_TASK_ID:?}"
fi
exec .venv/bin/python -u "$script" "$@"
