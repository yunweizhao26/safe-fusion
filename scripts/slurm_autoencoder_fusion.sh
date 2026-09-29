#!/usr/bin/env bash
#SBATCH --job-name=sf-autoencoder
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --array=0-7
#SBATCH --output=logs/slurm-autoencoder-%A_%a.out
#SBATCH --error=logs/slurm-autoencoder-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

units=(pancreas_0 pancreas_1 pancreas_2 colon)
modes=(resampled masked_positives)
unit_paths "${units[SLURM_ARRAY_TASK_ID / 2]}"
mode="${modes[SLURM_ARRAY_TASK_ID % 2]}"
extra=()
if [[ "${mode}" == masked_positives ]]; then
  mapfile -t extra < <(teacher_contract_args "${methods_root}")
  output="${methods_root}/autoencoder_fusion_5teachers"
else
  output="${methods_root}/autoencoder_fusion_3teachers"
fi
.venv/bin/python scripts/run_autoencoder_fusion.py --mode "${mode}" \
  --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
  --output "${output}" --seed 1729 "${extra[@]}"
