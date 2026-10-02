#!/bin/bash
#SBATCH --job-name=lb-masked-check
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=artifacts/paper_evidence/review_round4/label_baselines/masked/selfcheck-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=2
out=artifacts/paper_evidence/review_round4/label_baselines/masked
mkdir -p "$out/tmp"
export TMPDIR="$PWD/$out/tmp"
g++ -O3 -fopenmp -shared -fPIC scripts/analyses/label_baselines/masked/weighted_ap.cpp -o "$out/weighted_ap.so"
.venv/bin/python "scripts/analyses/label_baselines/masked/evaluate.py" --selfcheck
