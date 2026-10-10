#!/usr/bin/env bash
#SBATCH --job-name=sf-norman-supplement
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-norman-supplement-%x-%j.out
#SBATCH --error=logs/slurm-norman-supplement-%x-%j.err

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
data="${LF}/norman_crispra"
methods_root="${data}/methods"
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)

case "${STAGE}" in
  attribution)
    mapfile -t teacher_args < <(teacher_contract_args "${methods_root}")
    "${PY}" scripts/selector_attribution.py \
      --corrupted "${data}/corrupted.h5ad" --truth "${data}/prepared.h5ad" \
      --coordinates "${data}/coordinates.parquet" --splits "${data}/splits.parquet" \
      --fusion-contract "${methods_root}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${NR}/selector_mlp_attribution_range/norman_crispra" \
      --fit-split development --unit-column target \
      --variants full teacher_only context_only --architectures mlp --budgets "${BUDGETS[@]}" \
      --max-fit-rows 2000000 --bootstrap 2000 --seed 1729
    "${PY}" scripts/combine_selector_attribution.py \
      --input-dir "${NR}/selector_mlp_attribution_range/norman_crispra" \
      --output-dir "${NR}/selector_mlp_attribution_range/paired/norman_crispra" \
      --reference-selector full__mlp --bootstrap 2000 --seed 1729
    ;;
  benchmark_data)
    "${PY}" scripts/summarize_benchmark_data.py \
      --unit norman_crispra "${data}/corrupted.h5ad" "${data}/coordinates.parquet" "${data}/splits.parquet" \
      --output "${NR}/benchmark_data/benchmark_data.csv"
    ;;
  scgcl)
    .venv-baselines/bin/python scripts/run_scgcl_baseline.py \
      --corrupted "${data}/corrupted.h5ad" --output "${NR}/baselines/scgcl/norman" \
      --repo baselines_and_data/scGCL --epochs 300 --lr 1e-6 \
      --stability-note "common finite learning rate selected without masked truth" \
      --device cpu --checkpoint-every 25
    ;;
  scgcl_compare)
    out="${NR}/scgcl_comparison"
    "${PY}" scripts/write_leakage_free_units.py --root "${LF}" --keys norman_crispra \
      --contract scGCL "${NR}/baselines/scgcl/norman" --output "${out}/units_manifest.json"
    "${PY}" scripts/compute_matched_baseline_f1_curves.py --units-manifest "${out}/units_manifest.json" \
      --comparators scGCL \
      --output "${out}/scgcl_1000_points.csv" --summary-output "${out}/scgcl_summary.json" \
      --unit-output "${out}/masked_f1_unit_counts.parquet"
    "${PY}" scripts/paired_masked_f1_bootstrap.py --unit-counts "${out}/masked_f1_unit_counts.parquet" \
      --stacked-root "${out}/no_stacked_selectors" \
      --output "${out}/masked_f1_paired_bootstrap.csv" --summary-output "${out}/masked_f1_paired_bootstrap.json"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
