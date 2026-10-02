#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=$PWD/artifacts/paper_evidence/review_round4/sle_donor_labels/cache/matplotlib
export NUMBA_CACHE_DIR=$PWD/artifacts/paper_evidence/review_round4/sle_donor_labels/cache/numba
export XDG_CACHE_HOME=$PWD/artifacts/paper_evidence/review_round4/sle_donor_labels/cache
export TMPDIR=$PWD/artifacts/paper_evidence/review_round4/sle_donor_labels/cache
export WANDB_MODE=disabled
O=artifacts/paper_evidence/review_round4/sle_donor_labels
OLD=artifacts/paper_evidence/sle_treg_case
P=.venv/bin/python
EV=.venv-sle/bin/python
units=(main/fold_0/masked main/fold_0/deploy main/fold_1/masked main/fold_1/deploy main/fold_2/masked main/fold_2/deploy sex/fold_0/deploy sex/fold_1/deploy sex/fold_2/deploy cite/fold_0/deploy cite/fold_1/deploy cite/fold_2/deploy)
i=${SLURM_ARRAY_TASK_ID:-0}
if [[ $STAGE == verify_fills || $STAGE == chain || $STAGE == scvi || $STAGE == knn ]]; then
  unit=${units[i]}; set=${unit%%/*}; app=${unit##*/}; foldpart=${unit%/*}; fold=${foldpart##*_}
  if [[ $STAGE == verify_fills ]]; then
    $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$O/verification" sle_fills.py --set "$set" --fold "$fold" --app "$app"
    VALUE_VARIANT=verification $EV "scripts/analyses/sle_donor_labels/detection_fills.py" "$i"
    exit
  fi
  out=$O/conditional/$unit
  input=$out/hybrid.h5ad; splits=$out/splits.parquet
  if [[ $app == masked ]]; then input=$out/corrupted.h5ad; splits=$out/../splits.parquet; fi
  test -e "$O/verification/PASSED"
  if [[ $STAGE == knn ]]; then
    if [[ ! -e $out/teachers/graph_smooth/metadata.json ]]; then
      $P scripts/run_leakage_safe_method.py --method graph_smooth --transductive --input "$input" --coordinates "$out/coordinates.parquet" --splits "$splits" --output "$out/teachers/graph_smooth" --seed 1729 --condition-column donor
    fi
    exit
  fi
  if [[ $STAGE == scvi ]]; then
    if [[ ! -e $out/scvi_standard_donor/metadata.json ]]; then
      .conda-scvi-current/bin/python scripts/run_scvi_baseline.py --corrupted "$input" --coordinates "$out/coordinates.parquet" --splits "$splits" --output "$out/scvi_standard_donor" --epochs 200 --seed 1729 --condition-column donor
    fi
    $P scripts/count_scale_contract.py --contract "$out/scvi_standard_donor" --corrupted "$input" --output "$out/teachers/scvi"
    exit
  fi
  teachers=()
  for method in gene_median svd_impute graph_smooth magic scvi; do
    dest=$out/teachers/$method
    if [[ ! -e $dest/metadata.json ]]; then
      if [[ $method == graph_smooth ]]; then
        $P scripts/run_leakage_safe_method.py --method "$method" --transductive --input "$input" --coordinates "$out/coordinates.parquet" --splits "$splits" --output "$dest" --seed 1729 --condition-column donor
      else
        echo "Missing prerequisite teacher: $dest" >&2
        exit 1
      fi
    fi
    teachers+=(--teacher-contract "$dest")
  done
  if [[ ! -e $out/methods/safe_fusion/metadata.json ]]; then
    $P scripts/run_leakage_safe_method.py --method safe_fusion --transductive --input "$input" --coordinates "$out/coordinates.parquet" --splits "$splits" --output "$out/methods/safe_fusion" --seed 1729 "${teachers[@]}"
  fi
  if [[ ! -e $out/selector/calibration_report.json ]]; then
    mapfile -t genes < <($P -c 'import anndata,sys; print("\n".join(anndata.read_h5ad(sys.argv[1],backed="r").var_names))' "$out/../truth.h5ad")
    fill=(--budgets 0.01 0.05 0.1 --curve-points 2)
    if [[ $app == masked ]]; then fill=(--budgets 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 --curve-points 1000); fi
    $P "scripts/analyses/sle_donor_labels/selector.py" --corrupted "$input" --truth "$out/../truth.h5ad" --coordinates "$out/coordinates.parquet" --splits "$splits" --fusion-contract "$out/methods/safe_fusion" "${teachers[@]}" --output-dir "$out/selector" --fit-split development --architecture mlp --budget-mode apply_topk "${fill[@]}" --score-gene "${genes[@]}" --seed 1729 --detection-rule-mask-rate 0.1 --condition-column donor --condition-feature-source all_cells
  fi
  if [[ ! -e $out/fills/report.json ]]; then $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$O/conditional" sle_fills.py --set "$set" --fold "$fold" --app "$app"; fi
  $EV "scripts/analyses/sle_donor_labels/detection_fills.py" "$i"
  exit
fi
if [[ $STAGE == evaluate ]]; then
  case=$O/${VARIANT:?}; res=$case/results
  case $i in
    0)
      $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$case" sle_evaluate_masked.py --set main --output-dir "$res/masked"
      $EV scripts/paired_masked_f1_bootstrap.py --unit-counts "$res/masked/masked_f1_unit_counts.parquet" --stacked-root "$res/masked/no_stacked_selectors" --output "$res/masked/paired_bootstrap.csv" --summary-output "$res/masked/paired_bootstrap.json" --draws 2000 --seed 1729 ;;
    1) $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$case" sle_evaluate_zeros.py --set sex --output-dir "$res/q1_zeros_sex" ;;
    2) $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$case" sle_evaluate_zeros.py --set main --output-dir "$res/q1_zeros_main" ;;
    3) $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$case" sle_evaluate_deploy.py --set main --output-dir "$res/deploy_main" ;;
    4) $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$case" sle_evaluate_clustering.py --set main --output-dir "$res/q3_clustering" ;;
    5) $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$case" sle_evaluate_protein.py --set cite --adt "$case/cite/adt.parquet" --output-dir "$res/q7_protein" ;;
    6) $EV "scripts/analyses/sle_donor_labels/evaluate.py" "$case" sle_evaluate_focus_fills.py --output-dir "$res/focus_fills" ;;
  esac
fi
