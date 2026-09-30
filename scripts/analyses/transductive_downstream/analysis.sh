#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o artifacts/paper_evidence/review_round4/transductive_downstream/logs/%x-%A_%a.out
#SBATCH -e artifacts/paper_evidence/review_round4/transductive_downstream/logs/%x-%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
R=artifacts/paper_evidence/review_round4/transductive_downstream
if [[ "$1" == disease_endpoints ]]; then
  .venv/bin/python -u "scripts/analyses/transductive_downstream/disease_endpoints.py" "$SLURM_ARRAY_TASK_ID"
elif [[ "$1" == disease ]]; then
  .venv-scanpy/bin/python -u "scripts/analyses/transductive_downstream/disease.py" "$SLURM_ARRAY_TASK_ID" "$2"
else
  .venv/bin/python -u "scripts/analyses/transductive_downstream/$1.py" "${@:2}"
fi
