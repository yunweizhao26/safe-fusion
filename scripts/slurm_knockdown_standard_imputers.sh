#!/usr/bin/env bash
#SBATCH --job-name=sf-kd-standard
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-kd-standard-%A_%a.out
#SBATCH --error=logs/slurm-kd-standard-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh
source scripts/knockdown_paths.sh

screens=(norman_crispra adamson_crispri papalexi_eccite)
screen="${screens[SLURM_ARRAY_TASK_ID]}"
knockdown_paths "${screen}"
output="$(kd_standard_output "${METHOD:?METHOD must be alra, magic, scvi or saver}")"
mkdir -p "$(dirname "${output}")"

case "${METHOD}" in
  alra)
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    .venv/bin/python -u scripts/run_alra_baseline.py --corrupted "${out}/hybrid.h5ad" \
      --splits "${out}/splits.parquet" --output "${output}" \
      --components 100 --power-iterations 2 --quantile-prob 0.001 --seed 1729
    ;;
  magic)
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py --corrupted "${out}/hybrid.h5ad" \
      --coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet" \
      --output "${output}" --n-jobs "${SLURM_CPUS_PER_TASK}" --seed 1729
    ;;
  scvi)
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py --corrupted "${out}/hybrid.h5ad" \
      --coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet" \
      --output "${output}" --epochs 200 --seed 1729
    ;;
  saver)
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
    .venv/bin/python scripts/run_saver_baseline.py --corrupted "${out}/hybrid.h5ad" \
      --splits "${out}/splits.parquet" --output "${output}" --ncores "${SLURM_CPUS_PER_TASK}"
    ;;
  *)
    echo "unknown METHOD ${METHOD}" >&2
    exit 2
    ;;
esac
