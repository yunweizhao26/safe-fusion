#!/bin/bash
#SBATCH --job-name=main_tables_v2
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/main_tables_v2-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
C=scripts/analyses/main_tables_v2
mkdir -p artifacts/paper_evidence/review_round4/main_tables_v2
.venv/bin/python "$C/build_tables.py"
.venv/bin/python "$C/write_report.py"
.venv/bin/python "$C/verify_outputs.py"
