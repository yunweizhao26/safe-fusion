#!/usr/bin/env bash
#SBATCH --job-name=sf-colon-crossfit
#SBATCH --account=torch_pr_634_general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-colon-crossfit-%x-%A_%a.out
#SBATCH --error=logs/slurm-colon-crossfit-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | splits | data | units | teachers | scvi_teacher | stack | selector | alra | saver | magic | scvi | scgpt | stacked | nonzero_probability | tr_teachers | tr_stack | fv_selectors | evaluate}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
source scripts/unit_paths.sh

PY=.venv/bin/python
CF="${COLON_CROSSFIT_ROOT:-artifacts/paper_evidence/review_round3/colon_crossfit}"
MANIFEST="${CF}/units_manifest.json"
FV="${CF}/fusion_value"
FOLDS=(0 1 2)
UNIT_KEYS=(colon_0 colon_1 colon_2)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
BUDGETS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
task="${SLURM_ARRAY_TASK_ID:-0}"

fold_paths() {
  data="${CF}/fold_$1"
  input="${data}/corrupted.h5ad"
  coordinates="${data}/coordinates.parquet"
  splits="${data}/splits.parquet"
  truth="${data}/prepared.h5ad"
  selector_out="${data}/selector_mlp_biology_range_fullteachers"
  baseline="colon_fold_$1"
  key="colon_$1"
}

case "${stage}" in
  submit)
    mkdir -p logs
    sub() { sbatch --parsable "$@"; }
    splits_job=$(sub --job-name=sf-ccf-splits --time=00:20:00 --mem=8G --cpus-per-task=2 "${SCRIPT}" splits)
    data=$(sub --job-name=sf-ccf-data --array=0-2 --time=00:45:00 --mem=24G --dependency=afterok:${splits_job} "${SCRIPT}" data)
    units=$(sub --job-name=sf-ccf-units --time=00:10:00 --mem=4G --cpus-per-task=2 --dependency=afterok:${data} "${SCRIPT}" units)
    teachers=$(sub --job-name=sf-ccf-teach --array=0-11 --time=01:30:00 --dependency=afterok:${data} "${SCRIPT}" teachers)
    scvi_teacher=$(sub --job-name=sf-ccf-scvit --array=0-2 --gres=gpu:l40s:1 --dependency=afterok:${data} "${SCRIPT}" scvi_teacher)
    stack=$(sub --job-name=sf-ccf-stack --array=0-2 --mem=48G --dependency=afterok:${teachers}:${scvi_teacher} "${SCRIPT}" stack)
    selector=$(sub --job-name=sf-ccf-sel --array=0-2 --partition=cs --dependency=afterok:${stack} "${SCRIPT}" selector)
    alra=$(sub --job-name=sf-ccf-alra --array=0-2 --time=00:30:00 --mem=24G --dependency=afterok:${data} "${SCRIPT}" alra)
    saver=$(sub --job-name=sf-ccf-saver --array=0-2 --time=04:00:00 --dependency=afterok:${data} "${SCRIPT}" saver)
    magic=$(sub --job-name=sf-ccf-magic --array=0-2 --mem=48G --dependency=afterok:${data} "${SCRIPT}" magic)
    scvi=$(sub --job-name=sf-ccf-scvi --array=0-2 --mem=64G --gres=gpu:l40s:1 --dependency=afterok:${data} "${SCRIPT}" scvi)
    scgpt=$(sub --job-name=sf-ccf-scgpt --array=0-2 --mem=64G --gres=gpu:h200:1 --dependency=afterok:${data} "${SCRIPT}" scgpt)
    baselines="${teachers}:${scvi_teacher}:${alra}:${saver}:${magic}:${scvi}:${scgpt}"
    stacked=$(sub --job-name=sf-ccf-stk --array=0-2 --partition=cs --mem=48G --dependency=afterok:${units}:${baselines} "${SCRIPT}" stacked)
    probability=$(sub --job-name=sf-ccf-prob --array=0-5 --time=00:30:00 --dependency=afterok:${units}:${scvi}:${saver} "${SCRIPT}" nonzero_probability)
    tr_teachers=$(sub --job-name=sf-ccf-trt --array=0-14 --time=01:00:00 --dependency=afterok:${units}:${magic}:${scvi} "${SCRIPT}" tr_teachers)
    tr_stack=$(sub --job-name=sf-ccf-trs --array=0-2 --mem=48G --dependency=afterok:${tr_teachers} "${SCRIPT}" tr_stack)
    fv_selectors=$(sub --job-name=sf-ccf-fv --array=0-53 --partition=cs --mem=48G --dependency=afterok:${units}:${baselines} "${SCRIPT}" fv_selectors)
    fv_transductive=$(sub --job-name=sf-ccf-fvt --array=0-2 --partition=cs --mem=48G --export=ALL,FAMILIES=transductive --dependency=afterok:${tr_teachers} "${SCRIPT}" fv_selectors)
    evaluate=$(sub --job-name=sf-ccf-eval --time=01:30:00 --mem=64G --dependency=afterok:${selector}:${stacked}:${probability}:${fv_selectors}:${fv_transductive}:${tr_stack} "${SCRIPT}" evaluate)
    echo "splits=${splits_job} data=${data} units=${units} teachers=${teachers} scvi_teacher=${scvi_teacher} stack=${stack} selector=${selector}"
    echo "alra=${alra} saver=${saver} magic=${magic} scvi=${scvi} scgpt=${scgpt} stacked=${stacked} probability=${probability}"
    echo "tr_teachers=${tr_teachers} tr_stack=${tr_stack} fv_selectors=${fv_selectors} fv_transductive=${fv_transductive} evaluate=${evaluate}"
    ;;
  splits)
    "${PY}" scripts/make_colon_crossfit_splits.py --input external_data/prepared/colon_epithelial.h5ad \
      --output-dir "${CF}" --folds 3 --seed 1729
    ;;
  data)
    fold_paths "${FOLDS[task]}"
    "${PY}" scripts/prepare_colon_atlas.py \
      --input external_data/cellxgene/63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad \
      --output "${truth}" --report "${data}/prepared.report.json" --seed 1729 \
      --gene-selection-splits "${splits}"
    "${PY}" scripts/make_mask_replicate.py --truth "${truth}" --splits "${splits}" \
      --unit-column donor --seed 1729 --output-dir "${data}"
    ;;
  units)
    "${PY}" scripts/write_colon_crossfit_units.py --root "${CF}" --seed 1729
    ;;
  teachers)
    fold_paths "${FOLDS[task / 4]}"
    method="${CPU_TEACHERS[task % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${data}/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
        --output "${data}/${method}" --seed 1729
    fi
    ;;
  scvi_teacher)
    fold_paths "${FOLDS[task]}"
    .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${data}/scvi_inductive" --seed 1729
    ;;
  stack)
    fold_paths "${FOLDS[task]}"
    mapfile -t teacher_args < <(teacher_contract_args "${data}")
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${data}/safe_fusion" --value-model boosted --seed 1729 "${teacher_args[@]}"
    ;;
  selector)
    fold_paths "${FOLDS[task]}"
    mapfile -t teacher_args < <(teacher_contract_args "${data}")
    fit_cells=$("${PY}" -c "import pandas as pd; print(int(pd.read_parquet('${splits}')['split'].isin(['development', 'validation']).sum()))")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${input}" --truth "${truth}" --coordinates "${coordinates}" --splits "${splits}" \
      --fusion-contract "${data}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${selector_out}" --fit-split validation --fit-cells "${fit_cells}" \
      --architecture mlp --budget-mode apply_topk --budgets "${BUDGETS[@]}" \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729
    ;;
  alra)
    fold_paths "${FOLDS[task]}"
    export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
    "${PY}" scripts/run_alra_baseline.py --corrupted "${input}" --splits "${splits}" \
      --output "${CF}/baselines/alra/${baseline}" \
      --components 100 --power-iterations 2 --quantile-prob 0.001 --seed 1729
    ;;
  saver)
    fold_paths "${FOLDS[task]}"
    "${PY}" scripts/run_saver_baseline.py --corrupted "${input}" --splits "${splits}" \
      --output "${CF}/baselines/saver/${baseline}" --ncores "${OMP_NUM_THREADS}"
    ;;
  magic)
    fold_paths "${FOLDS[task]}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${CF}/baselines/magic/${baseline}" \
      --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    ;;
  scvi)
    fold_paths "${FOLDS[task]}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py --corrupted "${input}" \
      --coordinates "${coordinates}" --splits "${splits}" --output "${CF}/baselines/scvi/${baseline}" \
      --epochs 200 --seed 1729
    ;;
  scgpt)
    fold_paths "${FOLDS[task]}"
    .venv-baselines/bin/python scripts/run_scgpt_gene_prediction.py --corrupted "${input}" \
      --coordinates "${coordinates}" --ckpt-dir external_data/baselines/scgpt_human \
      --output "${CF}/baselines/scgpt_mvc/${baseline}" --batch-size 64 --gene-batch-size 512 \
      --device cuda --seed 1729
    ;;
  stacked)
    "${PY}" scripts/stacked_selector_baselines.py --unit "${UNIT_KEYS[task]}" \
      --units-manifest "${MANIFEST}" --output-root "${CF}/stacked_selector_baselines" --seed 1729
    ;;
  nonzero_probability)
    fold_paths "${FOLDS[task / 2]}"
    methods=(scvi saver)
    method="${methods[task % 2]}"
    .conda-scvi-current/bin/python scripts/nonzero_probability.py --method "${method}" \
      --contract "${CF}/baselines/${method}/${baseline}" --corrupted "${input}" \
      --output "${FV}/nonzero_probability/${method}/${key}"
    ;;
  tr_teachers)

    teachers=(gene_median svd_impute graph_smooth magic scvi)
    fold_paths "${FOLDS[task / 5]}"
    teacher="${teachers[task % 5]}"
    output="${FV}/transductive/${key}/${teacher}"
    case "${teacher}" in
      magic|scvi)
        "${PY}" scripts/count_scale_contract.py \
          --contract "${CF}/baselines/${teacher}/${baseline}" --corrupted "${input}" --output "${output}"
        ;;
      *)
        "${PY}" scripts/run_leakage_safe_method.py --method "${teacher}" --transductive \
          --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
          --output "${output}" --seed 1729
        ;;
    esac
    ;;
  tr_stack)
    fold_paths "${FOLDS[task]}"
    teacher_args=()
    for teacher in gene_median svd_impute graph_smooth magic scvi; do
      teacher_args+=(--teacher-contract "${FV}/transductive/${key}/${teacher}")
    done
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --output "${FV}/transductive/${key}/safe_fusion" --seed 1729 "${teacher_args[@]}"
    ;;
  fv_selectors)

    "${PY}" scripts/fusion_value_selectors.py --units-manifest "${MANIFEST}" --unit-keys "${UNIT_KEYS[@]}" \
      --output-root "${FV}/selectors" --transductive-root "${FV}/transductive" \
      --array-index "${task}" --families ${FAMILIES:-fusion leave_one_out architecture stacked} --seed 1729
    ;;
  evaluate)
    "${PY}" scripts/compute_matched_baseline_f1_curves.py --units-manifest "${MANIFEST}" \
      --comparators "Weighted kNN" SVD ALRA SAVER MAGIC scVI scGPT "MAGIC (inductive)" "scVI (inductive)" \
      --output "${CF}/selector_f1_fillrate_baselines_1000_points.csv" \
      --summary-output "${CF}/selector_f1_fillrate_baselines_summary.json" \
      --unit-output "${CF}/masked_f1_unit_counts.parquet"
    "${PY}" scripts/combine_mlp_baseline_curves.py --units-manifest "${MANIFEST}" \
      --stacked-root "${CF}/stacked_selector_baselines" \
      --baseline "${CF}/selector_f1_fillrate_baselines_1000_points.csv" \
      --output "${CF}/selector_f1_fillrate_mlp_baselines_1000_points.csv" \
      --summary-output "${CF}/selector_f1_fillrate_mlp_baselines_summary.json"
    "${PY}" scripts/paired_masked_f1_bootstrap.py --unit-counts "${CF}/masked_f1_unit_counts.parquet" \
      --stacked-root "${CF}/stacked_selector_baselines" \
      --output "${CF}/masked_f1_paired_bootstrap.csv" --summary-output "${CF}/masked_f1_paired_bootstrap.json"
    "${PY}" scripts/fusion_value_bootstrap.py --units-manifest "${MANIFEST}" --unit-keys "${UNIT_KEYS[@]}" \
      --selector-root "${FV}/selectors" --probability-root "${FV}/nonzero_probability" \
      --output-dir "${FV}/evaluation" --seed 1729
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
echo "elapsed_s=${SECONDS}"
