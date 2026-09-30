#!/usr/bin/env bash
#SBATCH --job-name=sf-r3-comparators
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/slurm-r3-comparators-%x-%A_%a.out
#SBATCH --error=logs/slurm-r3-comparators-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
source scripts/standard_imputers/r3_units.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
unit="${R3_UNITS[SLURM_ARRAY_TASK_ID]}"
r3_unit_paths "${unit}"

case "${STAGE:?STAGE must be kcluster, fit or scvi_zinb}" in
  kcluster)
    .venv-scanpy/bin/python scripts/standard_imputers/r3_kcluster.py --corrupted "${corrupted}" \
      --output "${R3_ROOT}/kcluster/${unit}.json" --seed 1729
    ;;
  fit)
    .venv/bin/python -u scripts/standard_imputers/run_r3_comparator.py --method "${METHOD:?METHOD required}" \
      --corrupted "${corrupted}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${R3_ROOT}/fits/${METHOD}/${unit}" --kcluster-json "${R3_ROOT}/kcluster/${unit}.json" \
      --ncores "${SLURM_CPUS_PER_TASK}" --seed 1729
    ;;
  scvi_zinb)

    .conda-scvi-current/bin/python scripts/standard_imputers/run_scvi_zinb.py --corrupted "${corrupted}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${R3_ROOT}/fits/scvi_zinb/${unit}" \
      --epochs 200 --seed 1729
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
