#!/usr/bin/env bash
#SBATCH --job-name=sf-value-accuracy
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-value-accuracy-%j.out
#SBATCH --error=logs/slurm-value-accuracy-%j.err

# Accuracy of the inserted value (Table 2, Supplementary Table S7). The units of
# Table 2 are the pancreas folds, colon and Norman CRISPRa of unit_paths.sh, with
# the value contracts of slurm_safe_fusion_stack.sh (VALUE_MODEL boosted and
# linear) and slurm_autoencoder_fusion.sh and the selectors of
# slurm_complete_downstream_selectors.sh. The replicate units are these five
# units, the mask replicates of slurm_seed_replicates.sh (stages stack and
# selector, with VALUE_MODEL boosted and linear) and the thinning units of
# slurm_thinning_benchmark.sh (stages selector and stack with VALUE_MODEL=linear).
# Outputs go to artifacts/paper_evidence/value_accuracy/.
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
source scripts/unit_paths.sh

REP=artifacts/paper_evidence/seed_replicates
T=artifacts/paper_evidence/thinning
SEEDS=(1730 1731 1732 1733)

# Selector output directory of slurm_complete_downstream_selectors.sh.
selector_dir() {
  case "$1" in
    colon) echo artifacts/paper_evidence/selector_mlp_biology_range/colon ;;
    norman_crispra) echo artifacts/paper_evidence/selector_mlp_biology_range_fullteachers/norman_crispra ;;
    pancreas_*) echo "${methods_root}/selector_mlp_biology_range_fullteachers" ;;
  esac
}

# Unit and data directory names of a dataset key in slurm_seed_replicates.sh.
replicate_unit() {
  case "$1" in
    norman_crispra) echo norman ;;
    *) echo "$1" ;;
  esac
}
replicate_data() {
  case "$1" in
    pancreas_*) echo pancreas ;;
    *) replicate_unit "$1" ;;
  esac
}

args=()
for key in pancreas_0 pancreas_1 pancreas_2 colon norman_crispra; do
  unit_paths "${key}"
  selector="$(selector_dir "${key}")"
  args+=(--unit "${key}" "${input}" "${coordinates}" "${splits}" "${methods_root}" "${selector}")
  args+=(--replicate "mask/seed_1729/${key}" "${input}" "${coordinates}" "${splits}" "${methods_root}" "${selector}")
  unit="$(replicate_unit "${key}")"
  for seed in "${SEEDS[@]}"; do
    data="${REP}/seed_${seed}/data/$(replicate_data "${key}")"
    out="${REP}/seed_${seed}/${unit}"
    args+=(--replicate "mask/seed_${seed}/${key}" "${data}/corrupted.h5ad" "${data}/coordinates.parquet"
      "${splits}" "${out}" "${out}/selector")
  done
done
for unit in colon_thinning_050 colon_thinning_025 pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2; do
  case "${unit}" in
    colon_*) unit_paths colon; data="${T}/data/${unit}" ;;
    pancreas_*) unit_paths "pancreas_${unit##*_}"; data="${T}/data/pancreas_thinning_050" ;;
  esac
  args+=(--replicate "thinning/${unit}" "${data}/corrupted.h5ad" "${data}/coordinates.parquet"
    "${splits}" "${T}/${unit}" "${T}/${unit}/selector")
done

.venv/bin/python scripts/evaluate_value_accuracy.py "${args[@]}" \
  --bootstrap 2000 --seed 1729 --output-dir artifacts/paper_evidence/value_accuracy
