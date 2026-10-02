#!/bin/bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
O=artifacts/paper_evidence/review_round4/current_tables
export TMPDIR=$PWD/$O/tmp/$SLURM_JOB_ID
mkdir -p "$TMPDIR"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh
keys=(colon pancreas_0 pancreas_1 pancreas_2 norman_crispra)
key=${keys[SLURM_ARRAY_TASK_ID % 5]}
deployment_paths "$key"
if (( SLURM_ARRAY_TASK_ID < 5 )); then
 data=(--corrupted "$input" --coordinates "$coordinates" --splits "$splits")
 models=$methods_root
 dest=$O/original/rule/masked/$key
else
 data=(--corrupted "$out/hybrid.h5ad" --coordinates "$out/coordinates.parquet" --splits "$out/splits.parquet")
 models=$deploy_methods
 dest=$O/original/rule/recorded/$key
fi
mapfile -t teacher_args < <(teacher_contract_args "$models")
.venv/bin/python scripts/calibrated_selective_fill.py "${data[@]}" --truth "$truth" --fusion-contract "$models/safe_fusion" "${teacher_args[@]}" --output-dir "$dest" "${fit[@]}" --architecture mlp --budget-mode apply_topk --budgets 0.05 --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --detection-rule-mask-rate 0.10 --seed 1729 > "$TMPDIR/report.stdout"
