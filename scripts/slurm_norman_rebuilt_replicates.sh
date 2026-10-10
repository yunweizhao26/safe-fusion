#!/usr/bin/env bash
#SBATCH --job-name=sf-norman-rep
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-norman-rep-%x-%A_%a.out
#SBATCH --error=logs/slurm-norman-rep-%x-%A_%a.err

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
REP="${NR}/seed_replicates"
TRUTH="${NR}/prepared/norman_crispra.h5ad"
SPLITS="${LF}/norman_crispra/splits.parquet"
SEEDS=(1730 1731 1732 1733)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
task="${SLURM_ARRAY_TASK_ID:-0}"

replicate_paths() {

  data="${REP}/seed_$1/data/norman"
  out="${REP}/seed_$1/norman"
}

case "${STAGE}" in
  mask)
    seed="${SEEDS[task]}"
    replicate_paths "${seed}"
    "${PY}" scripts/make_mask_replicate.py --truth "${TRUTH}" --splits "${SPLITS}" \
      --unit-column condition --seed "${seed}" --output-dir "${data}"
    ;;
  teachers)
    seed="${SEEDS[task / 4]}"
    method="${CPU_TEACHERS[task % 4]}"
    replicate_paths "${seed}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
        --splits "${SPLITS}" --output "${out}/${method}" \
        --seed "${seed}" --n-jobs "${SLURM_CPUS_PER_TASK}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
        --splits "${SPLITS}" --output "${out}/${method}" --seed "${seed}"
    fi
    ;;
  scvi_teacher)
    seed="${SEEDS[task]}"
    replicate_paths "${seed}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${SPLITS}" --output "${out}/scvi_inductive" --seed "${seed}"
    ;;
  stack)
    seed="${SEEDS[task]}"
    replicate_paths "${seed}"
    mapfile -t teacher_args < <(teacher_contract_args "${out}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${SPLITS}" --output "${out}/$(fused_value_contract "${VALUE_MODEL:-boosted}")" \
      --value-model "${VALUE_MODEL:-boosted}" --seed "${seed}" "${teacher_args[@]}"
    ;;
  selector)
    seed="${SEEDS[task]}"
    replicate_paths "${seed}"
    mapfile -t teacher_args < <(teacher_contract_args "${out}")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${data}/corrupted.h5ad" --truth "${TRUTH}" \
      --coordinates "${data}/coordinates.parquet" --splits "${SPLITS}" \
      --fusion-contract "${out}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${out}/selector" --fit-split development \
      --architecture mlp --budget-mode apply_topk --budgets 0.01 0.05 0.10 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
      --seed "${seed}"
    ;;
  magic)
    seed="${SEEDS[task]}"
    replicate_paths "${seed}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py \
      --corrupted "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${SPLITS}" --output "${REP}/seed_${seed}/baselines/magic/norman" \
      --n-jobs "${SLURM_CPUS_PER_TASK}" --seed "${seed}"
    ;;
  scvi)
    seed="${SEEDS[task]}"
    replicate_paths "${seed}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${data}/corrupted.h5ad" --coordinates "${data}/coordinates.parquet" \
      --splits "${SPLITS}" --output "${REP}/seed_${seed}/baselines/scvi/norman" \
      --epochs 200 --seed "${seed}"
    ;;
  summary)
    "${PY}" scripts/summarize_seed_replicates.py --evidence-root "${LF}" --replicate-root "${REP}" \
      --datasets CRISPRa --output-dir "${REP}/summary"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
