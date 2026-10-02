#!/bin/bash
#SBATCH --job-name=lb-score-replay
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-8
#SBATCH --output=artifacts/paper_evidence/review_round4/label_baselines/masked/replay-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_NUM_THREADS=8
out=artifacts/paper_evidence/review_round4/label_baselines/masked
mkdir -p "$out/tmp"
export TMPDIR="$PWD/$out/tmp"
sets=(adamson_crispri papalexi_eccite norman_crispra pancreas_0 pancreas_1 pancreas_2 colon_0 colon_1 colon_2)
.venv/bin/python "scripts/analyses/label_baselines/masked/replay.py" "${sets[$SLURM_ARRAY_TASK_ID]}"
