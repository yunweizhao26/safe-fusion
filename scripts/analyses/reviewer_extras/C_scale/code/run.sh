#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export SCALE_ROOT="$PWD/artifacts/paper_evidence/review_round4/reviewer_extras/C_scale"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SCALE_ROOT/cache" MPLCONFIGDIR="$SCALE_ROOT/cache/matplotlib" NUMBA_CACHE_DIR="$SCALE_ROOT/cache/numba"
export TORCH_HOME="$SCALE_ROOT/cache/torch" HF_HOME="$SCALE_ROOT/cache/huggingface"
export TMPDIR="$SCALE_ROOT/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$TORCH_HOME" "$HF_HOME"
export STAGE="$1" SLURM_ARRAY_TASK_ID="${2:-0}"
if [[ "$STAGE" == mask ]]; then .venv/bin/python "scripts/analyses/reviewer_extras/C_scale/code/check_prepared.py"; fi
exec bash "scripts/analyses/reviewer_extras/C_scale/code/stage.sh"
