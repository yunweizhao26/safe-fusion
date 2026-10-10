#!/usr/bin/env bash
#SBATCH --job-name=sle-pipe
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/slurm-sle-%x-%A_%a.out
#SBATCH --error=logs/slurm-sle-%x-%A_%a.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PY=.venv/bin/python
ROOT="${SLE_ROOT:-artifacts/paper_evidence/sle_treg_case}"
SOURCE_SHA256=3c0b74d54c03838a49817edce95314e6a4fa048ef35b74d1e697b8b2fd07cc03
CAP_PER_TYPE="${CAP_PER_TYPE:-100}"
SEED=1729
SETS=(main sex cite)
UNITS=(main/fold_0/masked main/fold_0/deploy main/fold_1/masked main/fold_1/deploy main/fold_2/masked main/fold_2/deploy
       sex/fold_0/deploy sex/fold_1/deploy sex/fold_2/deploy cite/fold_0/deploy cite/fold_1/deploy cite/fold_2/deploy)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
task="${SLURM_ARRAY_TASK_ID}"
step() { echo "step=$1 elapsed_s=${SECONDS}"; }

unit_paths() {

  set="${1%%/*}"
  fold_dir="${ROOT}/${1%/*}"
  app="${1##*/}"
  out="${fold_dir}/${app}"
  truth="${fold_dir}/truth.h5ad"
  if [[ "${app}" == masked ]]; then
    input="${out}/corrupted.h5ad"
    coordinates="${out}/coordinates.parquet"
    splits="${fold_dir}/splits.parquet"
  else
    input="${out}/hybrid.h5ad"
    coordinates="${out}/coordinates.parquet"
    splits="${out}/splits.parquet"
  fi
}

teacher_args() {
  local name
  for name in gene_median svd_impute graph_smooth magic_inductive scvi_inductive; do
    printf -- '--teacher-contract\n%s\n' "${out}/methods/${name}"
  done
}

case "${STAGE}" in
  prepare)
    set="${SETS[task]}"
    if [[ "${set}" == cite ]]; then
      "${PY}" scripts/sle_prepare_citeseq.py --output external_data/prepared/sle_cite.h5ad \
        --adt-output "${ROOT}/cite/adt.parquet" --fold-dir "${ROOT}/cite" \
        --cap-per-type "${CAP_PER_TYPE}" --folds 3 --seed "${SEED}"
      exit 0
    fi
    if [[ "${set}" == main ]]; then
      design=(--cohort 4.0 --contrast condition --donors-per-stratum 16)
    else
      design=(--cohort 2.0 --disease "systemic lupus erythematosus" --contrast sex --donors-per-stratum 8)
    fi
    "${PY}" scripts/sle_prepare.py --source-sha256 "${SOURCE_SHA256}" \
      --output "external_data/prepared/sle_${set}.h5ad" --fold-dir "${ROOT}/${set}" \
      --ancestry "European American" "${design[@]}" --cap-per-type "${CAP_PER_TYPE}" --folds 3 --seed "${SEED}"
    ;;
  build)
    set="${SETS[task / 3]}"
    fold_dir="${ROOT}/${set}/fold_$((task % 3))"
    "${PY}" scripts/sle_build_fold.py --prepared "external_data/prepared/sle_${set}.h5ad" \
      --splits "${fold_dir}/splits.parquet" --output "${fold_dir}/truth.h5ad"
    step genes
    "${PY}" scripts/make_mask_replicate.py --truth "${fold_dir}/truth.h5ad" --splits "${fold_dir}/splits.parquet" \
      --unit-column donor --seed "${SEED}" --output-dir "${fold_dir}/masked"
    step mask
    "${PY}" scripts/build_deployment_inputs.py --truth "${fold_dir}/truth.h5ad" \
      --corrupted "${fold_dir}/masked/corrupted.h5ad" --coordinates "${fold_dir}/masked/coordinates.parquet" \
      --splits "${fold_dir}/splits.parquet" --output-dir "${fold_dir}/deploy"
    step hybrid
    ;;
  teachers)
    unit_paths "${UNITS[task / 4]}"
    method="${CPU_TEACHERS[task % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${out}/methods/${method}" --seed "${SEED}" --n-jobs "${SLURM_CPUS_PER_TASK}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${out}/methods/${method}" --seed "${SEED}"
    fi
    step "${method}"
    ;;
  scvi_teacher)
    unit_paths "${UNITS[task]}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${out}/methods/scvi_inductive" --seed "${SEED}"
    step scvi_inductive
    ;;
  magic)
    unit_paths "${UNITS[task]}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${out}/baselines/magic" \
      --n-jobs "${SLURM_CPUS_PER_TASK}" --seed "${SEED}"
    step magic
    ;;
  scvi)
    unit_paths "${UNITS[task]}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${out}/baselines/scvi" \
      --epochs 200 --seed "${SEED}"
    step scvi
    ;;
  stack)
    unit_paths "${UNITS[task]}"
    mapfile -t teachers < <(teacher_args)
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${out}/methods/safe_fusion" --seed "${SEED}" "${teachers[@]}"
    step safe_fusion
    ;;
  selector)
    unit_paths "${UNITS[task]}"
    mapfile -t teachers < <(teacher_args)
    mapfile -t genes < <("${PY}" -c "import anndata, sys; print('\n'.join(anndata.read_h5ad(sys.argv[1], backed='r').var_names))" "${truth}")
    if [[ "${app}" == masked ]]; then
      fill=(--budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 --curve-points 1000)
    else
      fill=(--budgets 0.01 0.05 0.1 --curve-points 2)
    fi
    "${PY}" scripts/calibrated_selective_fill.py --corrupted "${input}" --truth "${truth}" \
      --coordinates "${coordinates}" --splits "${splits}" --fusion-contract "${out}/methods/safe_fusion" \
      "${teachers[@]}" --output-dir "${out}/selector" --fit-split development \
      --architecture mlp --budget-mode apply_topk "${fill[@]}" --score-gene "${genes[@]}" --seed "${SEED}"
    step selector
    ;;
  fills)
    unit_paths "${UNITS[task]}"
    fold="${fold_dir##*_}"
    .venv-sle/bin/python scripts/sle_fills.py --set "${set}" --fold "${fold}" --app "${app}"
    step fills
    ;;
  evaluate)
    EV=.venv-sle/bin/python
    RES="${ROOT}/results"
    case "${task}" in
      0)
        "${EV}" scripts/sle_evaluate_masked.py --set main --output-dir "${RES}/masked"
        "${EV}" scripts/paired_masked_f1_bootstrap.py --unit-counts "${RES}/masked/masked_f1_unit_counts.parquet" \
          --stacked-root "${RES}/masked/no_stacked_selectors" --output "${RES}/masked/paired_bootstrap.csv" \
          --summary-output "${RES}/masked/paired_bootstrap.json" --draws 2000 --seed "${SEED}"
        ;;
      1) "${EV}" scripts/sle_evaluate_zeros.py --set sex --output-dir "${RES}/q1_zeros_sex" ;;
      2) "${EV}" scripts/sle_evaluate_zeros.py --set main --output-dir "${RES}/q1_zeros_main" ;;
      3) "${EV}" scripts/sle_evaluate_deploy.py --set main --output-dir "${RES}/deploy_main" ;;
      4) "${EV}" scripts/sle_evaluate_clustering.py --set main --output-dir "${RES}/q3_clustering" ;;
      5) "${EV}" scripts/sle_evaluate_protein.py --set cite --adt "${ROOT}/cite/adt.parquet" --output-dir "${RES}/q7_protein" ;;
      6) "${EV}" scripts/sle_evaluate_focus_fills.py --output-dir "${RES}/focus_fills" ;;
    esac
    step "evaluate_${task}"
    ;;
  *)
    echo "unknown STAGE ${STAGE:-unset}" >&2
    exit 2
    ;;
esac
