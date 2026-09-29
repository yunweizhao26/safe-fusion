#!/usr/bin/env bash
#SBATCH --job-name=sf-norman-value
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-norman-value-%x-%A_%a.out
#SBATCH --error=logs/slurm-norman-value-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

PY=.venv/bin/python
LF="${LEAKAGE_FREE_ROOT:-artifacts/paper_evidence/review_round2/leakage_free}"
NR="${NORMAN_REBUILT_ROOT:-artifacts/paper_evidence/review_round2/norman_rebuilt}"
BENCH="${LF}/norman_crispra"
METHODS="${NR}/methods"
SELECTOR="${LF}/selector_mlp_biology_range_fullteachers/norman_crispra"
REP=artifacts/paper_evidence/seed_replicates
NREP="${NR}/seed_replicates"
T=artifacts/paper_evidence/thinning
SEEDS=(1730 1731 1732 1733)
task="${SLURM_ARRAY_TASK_ID:-0}"
input="${BENCH}/corrupted.h5ad"
coordinates="${BENCH}/coordinates.parquet"
splits="${BENCH}/splits.parquet"

case "${STAGE}" in
  linear)
    mkdir -p "${METHODS}"
    for name in "${TEACHERS[@]}" safe_fusion; do
      ln -sfn "$(pwd)/${BENCH}/methods/${name}" "${METHODS}/${name}"
    done
    mapfile -t teacher_args < <(teacher_contract_args "${METHODS}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${METHODS}/safe_fusion_linear" --value-model linear --seed 1729 "${teacher_args[@]}"
    ;;
  autoencoder)
    modes=(resampled masked_positives)
    mode="${modes[task]}"
    extra=()
    if [[ "${mode}" == masked_positives ]]; then
      mapfile -t extra < <(teacher_contract_args "${METHODS}")
      output="${METHODS}/autoencoder_fusion_5teachers"
    else
      output="${METHODS}/autoencoder_fusion_3teachers"
    fi
    "${PY}" scripts/run_autoencoder_fusion.py --mode "${mode}" \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${output}" --seed 1729 "${extra[@]}"
    ;;
  evaluate)
    args=()
    for key in pancreas_0 pancreas_1 pancreas_2 colon; do
      unit_paths "${key}"
      case "${key}" in
        colon) selector=artifacts/paper_evidence/selector_mlp_biology_range/colon; data_name=colon ;;
        *) selector="${methods_root}/selector_mlp_biology_range_fullteachers"; data_name=pancreas ;;
      esac
      args+=(--unit "${key}" "${input}" "${coordinates}" "${splits}" "${methods_root}" "${selector}")
      args+=(--replicate "mask/seed_1729/${key}" "${input}" "${coordinates}" "${splits}" "${methods_root}" "${selector}")
      for seed in "${SEEDS[@]}"; do
        data="${REP}/seed_${seed}/data/${data_name}"
        out="${REP}/seed_${seed}/${key}"
        args+=(--replicate "mask/seed_${seed}/${key}" "${data}/corrupted.h5ad" "${data}/coordinates.parquet"
          "${splits}" "${out}" "${out}/selector")
      done
    done
    input="${BENCH}/corrupted.h5ad"
    coordinates="${BENCH}/coordinates.parquet"
    splits="${BENCH}/splits.parquet"
    args+=(--unit norman_crispra "${input}" "${coordinates}" "${splits}" "${METHODS}" "${SELECTOR}")
    args+=(--replicate mask/seed_1729/norman_crispra "${input}" "${coordinates}" "${splits}" "${METHODS}" "${SELECTOR}")
    for seed in "${SEEDS[@]}"; do
      data="${NREP}/seed_${seed}/data/norman"
      out="${NREP}/seed_${seed}/norman"
      args+=(--replicate "mask/seed_${seed}/norman_crispra" "${data}/corrupted.h5ad" "${data}/coordinates.parquet"
        "${splits}" "${out}" "${out}/selector")
    done
    for unit in colon_thinning_050 colon_thinning_025 pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2; do
      case "${unit}" in
        colon_*) unit_paths colon; data="${T}/data/${unit}" ;;
        pancreas_*) unit_paths "pancreas_${unit##*_}"; data="${T}/data/pancreas_thinning_050" ;;
      esac
      args+=(--replicate "thinning/${unit}" "${data}/corrupted.h5ad" "${data}/coordinates.parquet"
        "${splits}" "${T}/${unit}" "${T}/${unit}/selector")
    done
    "${PY}" scripts/evaluate_value_accuracy.py "${args[@]}" \
      --bootstrap 2000 --seed 1729 --output-dir "${NR}/value_accuracy"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
