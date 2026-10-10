#!/usr/bin/env bash
#SBATCH --job-name=sf-r3-table1
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs,cpu_short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-r3-table1-%j.out
#SBATCH --error=logs/slurm-r3-table1-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

MANIFEST="${MANIFEST:-artifacts/paper_evidence/review_round2/leakage_free/units_manifest.json}"
COLON_CF="${COLON_CF:-artifacts/paper_evidence/review_round3/colon_crossfit}"

.venv/bin/python scripts/r3_downstream_table1_metrics.py --units-manifest "${MANIFEST}" --draws 2000 --seed 1729
echo "step=table1_metrics elapsed_s=${SECONDS}"

.venv/bin/python scripts/r3_downstream_table1_metrics.py --units-manifest "${COLON_CF}/units_manifest.json" \
  --unit-keys colon_0 colon_1 colon_2 --selector-root "${COLON_CF}/fusion_value/selectors" \
  --probability-root "${COLON_CF}/fusion_value/nonzero_probability" \
  --published "${COLON_CF}/masked_f1_paired_bootstrap.csv" \
  --output-dir artifacts/paper_evidence/review_round3/downstream/table1_metrics/colon_crossfit --draws 2000 --seed 1729
echo "step=table1_metrics_colon_crossfit elapsed_s=${SECONDS}"
.venv/bin/python scripts/r3_downstream_screen_f1.py
echo "step=screen_f1 elapsed_s=${SECONDS}"
