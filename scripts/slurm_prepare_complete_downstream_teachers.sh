#!/usr/bin/env bash
#SBATCH --job-name=sf-down-teachers
#SBATCH --time=00:35:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-11
#SBATCH --output=logs/slurm-complete-downstream-teachers-%A_%a.out
#SBATCH --error=logs/slurm-complete-downstream-teachers-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78
CF=artifacts/paper_evidence/pancreas_crossfit
NORMAN=artifacts/paper_evidence/norman_crispra

key=""
method="gene_median"
if (( SLURM_ARRAY_TASK_ID < 8 )); then
  keys=(pancreas_0 pancreas_1 pancreas_2 norman_crispra adamson_crispri dixit_ko papalexi_eccite zebrafish)
  key="${keys[SLURM_ARRAY_TASK_ID]}"
else
  keys=(adamson_crispri dixit_ko papalexi_eccite zebrafish)
  key="${keys[SLURM_ARRAY_TASK_ID-8]}"
  method="svd_impute"
fi

case "${key}" in
  pancreas_*)
    fold="${key##*_}"
    input="${PAN}/data/pancreas_islets/corrupted/mask_010.h5ad"
    coordinates="${PAN}/data/pancreas_islets/coordinates/mask_010.parquet"
    splits="${CF}/fold_${fold}/splits.parquet"
    output="${CF}/fold_${fold}/${method}"
    ;;
  norman_crispra)
    input="${NORMAN}/corrupted.h5ad"
    coordinates="${NORMAN}/coordinates.parquet"
    splits="${NORMAN}/splits.parquet"
    output="${NORMAN}/methods/${method}"
    ;;
  adamson_crispri|dixit_ko|papalexi_eccite)
    root="artifacts/external_perturbseq/${key}"
    input="${root}/corrupted.h5ad"
    coordinates="${root}/coordinates.parquet"
    splits="${root}/splits.parquet"
    output="${root}/${method}"
    ;;
  zebrafish)
    root=artifacts/external_trajectory/zebrafish
    input="${root}/corrupted.h5ad"
    coordinates="${root}/coordinates.parquet"
    splits="${root}/splits.parquet"
    output="${root}/${method}"
    ;;
  *)
    echo "unknown dataset ${key}" >&2
    exit 2
    ;;
esac

"${PY}" scripts/run_leakage_safe_method.py \
  --method "${method}" \
  --input "${input}" \
  --coordinates "${coordinates}" \
  --splits "${splits}" \
  --output "${output}" \
  --seed 1729
