#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
exec .venv/bin/python -u scripts/analyses/reviewer_extras/B_thinning/run.py fit "${SLURM_ARRAY_TASK_ID}"
