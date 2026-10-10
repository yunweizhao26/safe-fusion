#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
root="$PWD/artifacts/paper_evidence/review_round4/reviewer_extras/C_scale"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export XDG_CACHE_HOME="$root/cache" MPLCONFIGDIR="$root/cache/matplotlib" NUMBA_CACHE_DIR="$root/cache/numba"
export TMPDIR="$root/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
.venv/bin/python "scripts/analyses/reviewer_extras/C_scale/code/scale_evaluate.py" --root "$root/saver_snapshot" --sizes 200000 --draws 2000 --seed 1729 --method SAVER
date -u '+%FT%TZ' > "$root/saver_snapshot/completed_utc.txt"
