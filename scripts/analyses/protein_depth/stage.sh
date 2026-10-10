#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
O=$PWD/artifacts/paper_evidence/review_round4/protein_depth
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XDG_CACHE_HOME=$O/cache MPLCONFIGDIR=$O/cache/matplotlib NUMBA_CACHE_DIR=$O/cache/numba
export TMPDIR=$O/cache/tmp/${SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$O/tables" "$O/bootstrap"
start=$(date +%s)
trap 'code=$?; printf "%s,%s,%s,%s\n" "${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}" "$1" "$(( $(date +%s)-start ))" "$code" > "$O/runtime/job_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.csv"' EXIT
case $1 in
 prepare) export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8; .venv/bin/python -B "scripts/analyses/protein_depth/prepare.py" "$2"; .venv/bin/python -B "scripts/analyses/protein_depth/prepare.py" "audit_$2" ;;
 merge) .venv/bin/python -B "scripts/analyses/protein_depth/prepare.py" merge; .venv/bin/python -B scripts/analyses/protein_depth/evaluate.py selftest ;;
 pair) .venv/bin/python -B "scripts/analyses/protein_depth/evaluate.py" pair ;;
 original) .venv/bin/python -B "scripts/analyses/protein_depth/evaluate.py" original-tables "$2" ;;
 aggregate) .venv/bin/python -B "scripts/analyses/protein_depth/aggregate.py" ;;
 audit) .venv/bin/python -B "scripts/analyses/protein_depth/render_tables.py"; .venv/bin/python -B "scripts/analyses/protein_depth/audit.py" ;;
 report) .venv/bin/python -B "scripts/analyses/protein_depth/aggregate.py"; .venv/bin/python -B "scripts/analyses/protein_depth/render_tables.py"; .venv/bin/python -B "scripts/analyses/protein_depth/audit.py"; .venv/bin/python -B scripts/analyses/protein_depth/verify_observation.py; .venv/bin/python -B "scripts/analyses/protein_depth/report.py" ;;
 *) exit 2 ;;
esac
