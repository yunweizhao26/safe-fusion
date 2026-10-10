#!/usr/bin/env bash
#SBATCH --job-name=s20-added-rankings
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --array=0-1
#SBATCH --output=artifacts/paper_evidence/review_round4/protein_s20_rows/logs/%A_%a.out
#SBATCH --error=artifacts/paper_evidence/review_round4/protein_s20_rows/logs/%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
OUT=artifacts/paper_evidence/review_round4/protein_s20_rows
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$PWD/$OUT/runtime/tmp_${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="$TMPDIR/cache" MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR"
if [[ "$SLURM_ARRAY_TASK_ID" == 0 ]]; then
    .venv/bin/python "scripts/analyses/protein_s20_rows/evaluate.py" --block small --include-p-nonzero --include-magic --workers 3
else
    .venv/bin/python "scripts/analyses/protein_s20_rows/evaluate.py" --block full --include-p-nonzero --workers 3
fi
