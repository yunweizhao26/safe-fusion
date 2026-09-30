#!/usr/bin/env bash
#SBATCH --job-name=sf-leakage-free
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-leakage-free-%x-%A_%a.out
#SBATCH --error=logs/slurm-leakage-free-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh

PY=.venv/bin/python
LF="${LEAKAGE_FREE_ROOT:-artifacts/paper_evidence/review_round2/leakage_free}"
LF_UNITS=(pancreas_0 pancreas_1 pancreas_2 norman_crispra)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
MANIFEST="${LF}/units_manifest.json"
task="${SLURM_ARRAY_TASK_ID:-0}"

leakage_free_paths() {

  case "$1" in
    pancreas_*)
      data="${LF}/pancreas_crossfit/fold_${1##*_}"
      methods_root="${data}"
      selector_out="${data}/selector_mlp_biology_range_fullteachers"
      baseline="pancreas_fold_${1##*_}"
      ;;
    norman_crispra)
      data="${LF}/norman_crispra"
      methods_root="${data}/methods"
      selector_out="${LF}/selector_mlp_biology_range_fullteachers/norman_crispra"
      baseline=norman
      ;;
    *)
      echo "unknown unit $1" >&2
      return 2
      ;;
  esac
  input="${data}/corrupted.h5ad"
  coordinates="${data}/coordinates.parquet"
  splits="${data}/splits.parquet"
  truth="${data}/prepared.h5ad"
}

case "${STAGE}" in
  data)
    if [[ "${task}" == 0 ]]; then

      "${PY}" scripts/make_pancreas_crossfit_splits.py \
        --input external_data/prepared/pancreas_islets.h5ad \
        --output-dir "${LF}/pancreas_crossfit" --folds 3 --seed 1729
      for fold in 0 1 2; do
        leakage_free_paths "pancreas_${fold}"
        "${PY}" scripts/prepare_pancreas_atlas.py \
          --input external_data/cellxgene/f89a618b-fe4b-404e-bd39-7c574529b1f5.h5ad \
          --output "${truth}" --report "${data}/prepared.report.json" --seed 1729 \
          --gene-selection-splits "${splits}"
        "${PY}" scripts/make_mask_replicate.py --truth "${truth}" --splits "${splits}" \
          --unit-column donor --seed 1729 --output-dir "${data}"
      done
      "${PY}" scripts/write_leakage_free_units.py --root "${LF}"

      mkdir -p "${LF}/selector_mlp_biology_range" "${LF}/stacked_selector_baselines"
      ln -sfn "$(pwd)/artifacts/paper_evidence/selector_mlp_biology_range/colon" "${LF}/selector_mlp_biology_range/colon"
      ln -sfn "$(pwd)/artifacts/paper_evidence/stacked_selector_baselines/colon" "${LF}/stacked_selector_baselines/colon"
    else
      leakage_free_paths norman_crispra
      "${PY}" scripts/prepare_norman_crispra.py --input "${NORMAN_SOURCE:?set NORMAN_SOURCE to perturb_processed.h5ad}" \
        --output "${truth}" --seed 1729 --max-cells-per-condition 60 --max-control-cells 1000 \
        --min-cells-per-condition 60 --target-log2fc-min 0.5 --target-p-max 0.05 --test-fraction 0.30
      "${PY}" scripts/setup_norman_crispra_experiment.py --input "${truth}" --output-dir "${data}" \
        --split-column preassigned_split --mask-fraction 0.10 --seed 1729
    fi
    ;;
  teachers)
    leakage_free_paths "${LF_UNITS[task / 4]}"
    method="${CPU_TEACHERS[task % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${methods_root}/${method}" --seed 1729 --n-jobs "${SLURM_CPUS_PER_TASK}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${methods_root}/${method}" --seed 1729
    fi
    ;;
  scvi_teacher)
    leakage_free_paths "${LF_UNITS[task]}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${methods_root}/scvi_inductive" --seed 1729
    ;;
  stack)
    leakage_free_paths "${LF_UNITS[task]}"
    mapfile -t teacher_args < <(teacher_contract_args "${methods_root}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${methods_root}/safe_fusion" --value-model boosted --seed 1729 "${teacher_args[@]}"
    ;;
  selector)
    leakage_free_paths "${LF_UNITS[task]}"
    mapfile -t teacher_args < <(teacher_contract_args "${methods_root}")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${input}" --truth "${truth}" --coordinates "${coordinates}" --splits "${splits}" \
      --fusion-contract "${methods_root}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${selector_out}" --fit-split development \
      --architecture mlp --budget-mode apply_topk --budgets "${BUDGETS[@]}" \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729
    ;;
  alra)
    leakage_free_paths "${LF_UNITS[task]}"
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    "${PY}" scripts/run_alra_baseline.py --corrupted "${input}" --splits "${splits}" \
      --output "${LF}/baselines/alra/${baseline}" \
      --components 100 --power-iterations 2 --quantile-prob 0.001 --seed 1729
    ;;
  saver)
    leakage_free_paths "${LF_UNITS[task]}"
    "${PY}" scripts/run_saver_baseline.py --corrupted "${input}" --splits "${splits}" \
      --output "${LF}/baselines/saver/${baseline}" --ncores "${SLURM_CPUS_PER_TASK}"
    ;;
  magic)
    leakage_free_paths "${LF_UNITS[task]}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${LF}/baselines/magic/${baseline}" \
      --n-jobs "${SLURM_CPUS_PER_TASK}" --seed 1729
    ;;
  scvi)
    leakage_free_paths "${LF_UNITS[task]}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${LF}/baselines/scvi/${baseline}" \
      --epochs 200 --seed 1729
    ;;
  scgpt)
    leakage_free_paths "${LF_UNITS[task]}"
    .venv-baselines/bin/python scripts/run_scgpt_gene_prediction.py --corrupted "${input}" \
      --coordinates "${coordinates}" --ckpt-dir external_data/baselines/scgpt_human \
      --output "${LF}/baselines/scgpt_mvc/${baseline}" --batch-size 64 --gene-batch-size 512 \
      --device cuda --seed 1729
    ;;
  stacked)
    "${PY}" scripts/stacked_selector_baselines.py --unit "${LF_UNITS[task]}" \
      --units-manifest "${MANIFEST}" --output-root "${LF}/stacked_selector_baselines" --seed 1729
    ;;
  evaluate)
    "${PY}" scripts/compute_matched_baseline_f1_curves.py --units-manifest "${MANIFEST}" \
      --comparators "Weighted kNN" SVD ALRA SAVER MAGIC scVI scGPT "MAGIC (inductive)" "scVI (inductive)" \
      --output "${LF}/selector_f1_fillrate_baselines_1000_points.csv" \
      --summary-output "${LF}/selector_f1_fillrate_baselines_summary.json" \
      --unit-output "${LF}/masked_f1_unit_counts.parquet"
    "${PY}" scripts/combine_mlp_baseline_curves.py --evidence-root "${LF}" \
      --stacked-root "${LF}/stacked_selector_baselines" \
      --baseline "${LF}/selector_f1_fillrate_baselines_1000_points.csv" \
      --output "${LF}/selector_f1_fillrate_mlp_baselines_1000_points.csv" \
      --summary-output "${LF}/selector_f1_fillrate_mlp_baselines_summary.json"
    "${PY}" scripts/paired_masked_f1_bootstrap.py --unit-counts "${LF}/masked_f1_unit_counts.parquet" \
      --stacked-root "${LF}/stacked_selector_baselines" \
      --output "${LF}/masked_f1_paired_bootstrap.csv" --summary-output "${LF}/masked_f1_paired_bootstrap.json"
    MPLBACKEND=Agg MPLCONFIGDIR="/tmp/sf-leakage-free-${SLURM_JOB_ID:-local}" \
      "${PY}" scripts/plot_selector_f1_fillrate.py \
      --input "${LF}/selector_f1_fillrate_mlp_baselines_1000_points.csv" \
      --output "${LF}/figures/f1_fillrate_3panel_oup.png"
    "${PY}" scripts/compare_leakage_free_benchmarks.py --root "${LF}"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
