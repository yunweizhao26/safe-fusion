#!/usr/bin/env bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
O=artifacts/paper_evidence/review_round4/transductive_comparators
export TMPDIR="$PWD/$O/tmp/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
if [[ "$1" == report ]]; then
    exec .venv/bin/python "scripts/analyses/transductive_comparators/report.py"
fi
exec .venv/bin/python "scripts/analyses/transductive_comparators/workflow.py" "$@"
