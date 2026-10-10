#!/usr/bin/env bash
#SBATCH --job-name=sf-r3-thin
#SBATCH --account=torch_pr_634_general
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-r3-thin-%x-%A_%a.out
#SBATCH --error=logs/slurm-r3-thin-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage: submit | prepare | cpu | gpu | selector | fill | evaluate | summarize}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

PY=.venv/bin/python
OUT="${THINNING_REFERENCE_ROOT:-artifacts/paper_evidence/review_round3/downstream/thinning_reference}"
TT=artifacts/paper_evidence/review_round2/thinning_transfer
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial
PAN=artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets
CF=artifacts/paper_evidence/pancreas_crossfit
NORMAN=artifacts/paper_evidence/review_round2/leakage_free/norman_crispra
ZF=artifacts/external_trajectory/zebrafish
UNITS=(colon_thinning_050 pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2 norman_thinning_050 zebrafish_thinning_050)
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
TEACHERS=(gene_median svd_impute graph_smooth magic_inductive scvi_inductive)
FRACTIONS=(0.01 0.05 0.1)
PCTS=(1 5 10)
SUFFIXES=(0p01 0p05 0p1)

unit_paths() {
  local key="$1"
  case "${key}" in
    colon_thinning_050)
      models="${TT}/mask_trained/${key}"; thinned=artifacts/paper_evidence/thinning/data/${key}/corrupted.h5ad
      truth="${COL}/preprocessed.h5ad"
      ;;
    pancreas_thinning_050_*)
      models="${TT}/mask_trained/${key}"; thinned=artifacts/paper_evidence/thinning/data/pancreas_thinning_050/corrupted.h5ad
      truth="${PAN}/preprocessed.h5ad"
      ;;
    norman_thinning_050)
      models="${TT}/mask_trained/${key}"; thinned="${TT}/data/${key}/corrupted.h5ad"
      truth="${NORMAN}/prepared.h5ad"
      ;;
    zebrafish_thinning_050)
      models="${OUT}/models/${key}"; thinned="${OUT}/data/${key}/corrupted.h5ad"
      truth=external_data/prepared/zebrafish_trajectory.h5ad
      ;;
    *) echo "unknown unit ${key}" >&2; return 2 ;;
  esac
  hybrid="${models}/input/hybrid.h5ad"
  hybrid_coordinates="${models}/input/coordinates.parquet"
  splits="${models}/input/splits.parquet"
  unit_dir="${OUT}/units/${key}"
}

zebrafish_teacher() {
  local method="$1"
  if [[ "${method}" == magic_inductive ]]; then
    .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
      --input "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
      --output "${models}/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
  else
    "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
      --input "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
      --output "${models}/${method}" --seed 1729
  fi
}

case "${stage}" in
  submit)
    mkdir -p logs
    prep=$(sbatch --parsable --job-name=sf-r3-thin-prep --mem=16G --time=00:30:00 "${SCRIPT}" prepare)
    cpu=$(sbatch --parsable --job-name=sf-r3-thin-cpu --array=0-4 --dependency=afterok:${prep} "${SCRIPT}" cpu)
    gpu=$(sbatch --parsable --job-name=sf-r3-thin-gpu --array=0-1 --gres=gpu:l40s:1 --dependency=afterok:${prep} "${SCRIPT}" gpu)
    sel=$(sbatch --parsable --job-name=sf-r3-thin-sel -p cs --mem=48G --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" selector)
    fill=$(sbatch --parsable --job-name=sf-r3-thin-fill --array=0-5 --mem=48G --time=01:00:00 --dependency=afterok:${sel} "${SCRIPT}" fill)
    eval=$(sbatch --parsable --job-name=sf-r3-thin-eval --array=0-3 --mem=64G --time=06:00:00 --dependency=afterok:${fill} "${SCRIPT}" evaluate)
    summ=$(sbatch --parsable --job-name=sf-r3-thin-summ --mem=48G --time=02:00:00 --dependency=afterok:${eval} "${SCRIPT}" summarize)
    echo "prepare=${prep} cpu=${cpu} gpu=${gpu} selector=${sel} fill=${fill} evaluate=${eval} summarize=${summ}"
    ;;
  prepare)
    unit_paths zebrafish_thinning_050
    "${PY}" scripts/prepare_thinning_transfer.py thin --truth "${truth}" --splits "${ZF}/splits.parquet" \
      --unit-column condition --retained-fraction 0.5 --seed 1729 --output-dir "$(dirname "${thinned}")"
    "${PY}" scripts/prepare_thinning_transfer.py hybrid --thinned-dir "$(dirname "${thinned}")" \
      --splits "${ZF}/splits.parquet" --unit-column condition --mask-seed 1729 --output-dir "${models}/input"
    ;;
  cpu)

    unit_paths zebrafish_thinning_050
    id="${SLURM_ARRAY_TASK_ID}"
    if (( id < 4 )); then
      zebrafish_teacher "${CPU_TEACHERS[id]}"
    else
      .conda-magic-current/bin/python scripts/run_magic_baseline.py \
        --corrupted "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${models}/magic" --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    fi
    echo "elapsed_s=${SECONDS}"
    ;;
  gpu)

    unit_paths zebrafish_thinning_050
    if (( SLURM_ARRAY_TASK_ID == 0 )); then
      .conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi \
        --input "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${models}/scvi_inductive" --seed 1729
    else
      .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
        --corrupted "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
        --output "${models}/scvi" --epochs 200 --seed 1729
    fi
    echo "elapsed_s=${SECONDS}"
    ;;
  selector)

    unit_paths zebrafish_thinning_050
    teacher_args=()
    for name in "${TEACHERS[@]}"; do teacher_args+=(--teacher-contract "${models}/${name}"); done
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion \
      --input "${hybrid}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
      --output "${models}/safe_fusion" --seed 1729 "${teacher_args[@]}"
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${hybrid}" --truth "${truth}" --coordinates "${hybrid_coordinates}" --splits "${splits}" \
      --fusion-contract "${models}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${models}/selector" --fit-split development \
      --architecture mlp --budget-mode apply_topk --budgets "${FRACTIONS[@]}" \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729 > /dev/null
    echo "elapsed_s=${SECONDS}"
    ;;
  fill)

    unit_paths "${UNITS[SLURM_ARRAY_TASK_ID]}"
    mkdir -p "${unit_dir}"
    for name in magic scvi; do
      "${PY}" scripts/count_scale_contract.py --contract "${models}/${name}" --corrupted "${hybrid}" \
        --output "${unit_dir}/models/${name}_counts"
    done
    for spec in "${models}/svd_impute svd" "${models}/graph_smooth weighted_knn" \
                "${unit_dir}/models/magic_counts magic" "${unit_dir}/models/scvi_counts scvi"; do
      set -- ${spec}
      "${PY}" scripts/apply_fill_fraction.py --corrupted "${hybrid}" --splits "${splits}" \
        --method-contract "$1" --method-name "$2" --output-root "${unit_dir}/matched" --fractions "${FRACTIONS[@]}"
    done
    sources=()
    if [[ "${UNITS[SLURM_ARRAY_TASK_ID]}" == pancreas_* ]]; then

      sources+=(--source "graph_smooth=${models}/graph_smooth" --source "safe_fusion=${models}/safe_fusion")
    fi
    for i in 0 1 2; do
      pct="${PCTS[i]}"; suffix="${SUFFIXES[i]}"
      sources+=(--source "safe_fusion_${pct}pct=${models}/selector/safe_fusion_calibrated_mlp_topk_${suffix}")
      for name in svd weighted_knn magic scvi; do
        sources+=(--source "${name}_${pct}pct=${unit_dir}/matched/${name}_topk_${suffix}")
      done
    done
    "${PY}" scripts/finalize_deployment_contracts.py --recorded "${thinned}" --splits "${splits}" \
      "${sources[@]}" --output-root "${unit_dir}"
    cp "${splits}" "${unit_dir}/splits.parquet"
    "${PY}" -c "import pandas as pd, sys; pd.read_parquet(sys.argv[1]).iloc[:0].to_parquet(sys.argv[2], index=False)" \
      "${hybrid_coordinates}" "${unit_dir}/empty_coordinates.parquet"
    echo "elapsed_s=${SECONDS}"
    ;;
  evaluate)
    methods_for() {
      local unit="$1" pct name
      methods=()
      for pct in "${PCTS[@]}"; do
        for name in safe_fusion svd weighted_knn magic scvi; do
          methods+=(--method "${name}_${pct}pct=${unit}/${name}_${pct}pct")
        done
      done
    }
    case "${SLURM_ARRAY_TASK_ID}" in
      0)
        unit_paths colon_thinning_050
        methods_for "${unit_dir}"
        "${PY}" scripts/evaluate_colon_donor_biology.py --truth "${truth}" --corrupted "${thinned}" \
          --coordinates "${unit_dir}/empty_coordinates.parquet" --splits "${unit_dir}/splits.parquet" \
          "${methods[@]}" --output-dir "${OUT}/evaluation/markers/colon" --bootstrap 2000 --seed 1729 \
          --marker-panel source --allow-transductive
        ;;
      1)
        extras=()
        for pct in "${PCTS[@]}"; do
          for name in safe_fusion svd weighted_knn magic scvi; do
            extras+=(--extra-method "${name}_${pct}pct=${name}_${pct}pct")
          done
        done

        mkdir -p "${OUT}/units/pancreas_crossfit"
        for fold in 0 1 2; do
          ln -sfn "$(realpath "${OUT}/units/pancreas_thinning_050_${fold}")" "${OUT}/units/pancreas_crossfit/fold_${fold}"
        done
        unit_paths pancreas_thinning_050_0
        "${PY}" scripts/evaluate_pancreas_crossfit_biology.py --truth "${truth}" --corrupted "${thinned}" \
          --coordinates "${unit_dir}/empty_coordinates.parquet" --crossfit-dir "${OUT}/units/pancreas_crossfit" \
          --safe-fusion-subdir safe_fusion "${extras[@]}" --output-dir "${OUT}/evaluation/pancreas_biology" \
          --bootstrap 2000 --seed 1729 --marker-panel source --allow-transductive
        ;;
      2)
        unit_paths zebrafish_thinning_050
        methods_for "${unit_dir}"
        "${PY}" scripts/evaluate_trajectory_preservation.py --truth "${truth}" --corrupted "${thinned}" \
          --splits "${unit_dir}/splits.parquet" "${methods[@]}" \
          --output-dir "${OUT}/evaluation/trajectory/zebrafish" --bootstrap 2000 --seed 1729 --allow-transductive
        ;;
      3)
        unit_paths norman_thinning_050
        methods_for "${unit_dir}"
        "${PY}" scripts/evaluate_interventional_grn.py --dataset norman_crispra --intervention gain_of_function \
          --truth "${truth}" --corrupted "${thinned}" --splits "${unit_dir}/splits.parquet" "${methods[@]}" \
          --publication-doi 10.1126/science.aax4438 --output-dir "${OUT}/evaluation/grn/norman_crispra" \
          --bootstrap 2000 --seed 1729 --allow-transductive
        ;;
    esac
    echo "elapsed_s=${SECONDS}"
    ;;
  summarize)
    "${PY}" scripts/r3_downstream_thinning.py --root "${OUT}" --draws 2000 --seed 1729
    echo "elapsed_s=${SECONDS}"
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
