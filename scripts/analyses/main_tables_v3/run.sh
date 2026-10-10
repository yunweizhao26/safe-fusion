#!/bin/bash
#SBATCH --job-name=main_tables_v3
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/main_tables_v3-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C=scripts/analyses/main_tables_v3
mkdir -p artifacts/paper_evidence/review_round4/main_tables_v3
 .venv/bin/python "$C/prepare_added_counts.py"
.venv/bin/python "$C/build_tables.py"
.venv/bin/python "$C/audit_zero_availability.py"
.venv/bin/python "$C/write_report.py"
.venv/bin/python "$C/verify_outputs.py"
