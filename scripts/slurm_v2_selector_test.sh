#!/usr/bin/env bash
#SBATCH --job-name=sf-v2-test
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/slurm-v2-test-%x-%A_%a.out
#SBATCH --error=logs/slurm-v2-test-%x-%A_%a.err

set -euo pipefail

stage="${1:?stage}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/$(basename -- "${BASH_SOURCE[0]}")"
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

PY=.venv/bin/python
V2=artifacts/paper_evidence/review_round3/selector_v2
TEST="${V2}/test"
LF=artifacts/paper_evidence/review_round2/leakage_free
TT=artifacts/paper_evidence/review_round2/thinning_transfer
TH=artifacts/paper_evidence/thinning
DEP=artifacts/paper_evidence/downstream_deployment
NKD=artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt/deployment
COL=artifacts/colon_runs/0b2469810675-c0db6f963e94
CCF=artifacts/paper_evidence/review_round3/colon_crossfit
PAPALEXI=artifacts/paper_evidence/papalexi_crossmodal/benchmark
CPU_TEACHERS=(gene_median svd_impute graph_smooth magic_inductive)
TRANS_CPU=(gene_median svd_impute graph_smooth)
TRANS_CHAINS=(0 1 2 3 4 19 20 21)
task="${SLURM_ARRAY_TASK_ID:-0}"

read -r design rho features slug < <("${PY}" -c "
import json, re
variant = json.load(open('${V2}/inner/evaluation_colon_crossfit/selection.json'))['chosen']['variant']
target, features = variant.split('|')
match = re.fullmatch(r'(.+)_0p(\d+)', target)
design, rho = (match.group(1), '0.' + match.group(2)) if match else (target, '0')
print(design, rho, features, target)
")

chain_paths() {
  local i="$1" fold
  layer=counts; fit_split=development; current=""
  if (( i < 5 )); then
    group=masked
    key=$(echo pancreas_0 pancreas_1 pancreas_2 colon norman_crispra | cut -d' ' -f$((i + 1)))
    case "${key}" in
      pancreas_*) local d="${LF}/pancreas_crossfit/fold_${key##*_}"
                  input="${d}/corrupted.h5ad"; coordinates="${d}/coordinates.parquet"; splits="${d}/splits.parquet"; recorded="${d}/prepared.h5ad" ;;
      colon) local d="${COL}/data/colon_epithelial"
             input="${d}/corrupted/mask_010.h5ad"; coordinates="${d}/coordinates/mask_010.parquet"; splits="${d}/splits.parquet"
             recorded="${d}/preprocessed.h5ad"; fit_split=validation ;;
      norman_crispra) local d="${LF}/norman_crispra"
                      input="${d}/corrupted.h5ad"; coordinates="${d}/coordinates.parquet"; splits="${d}/splits.parquet"; recorded="${d}/prepared.h5ad" ;;
    esac
  elif (( i < 11 )); then
    group=thinned
    key=$(echo colon_thinning_050 colon_thinning_025 pancreas_thinning_050_0 pancreas_thinning_050_1 pancreas_thinning_050_2 norman_thinning_050 | cut -d' ' -f$((i - 4)))
    local m="${TT}/mask_trained/${key}/input"
    input="${m}/hybrid.h5ad"; coordinates="${m}/coordinates.parquet"; splits="${m}/splits.parquet"; layer=corrupted_counts
    case "${key}" in
      colon_*) recorded="${TH}/data/${key}/corrupted.h5ad"; fit_split=validation ;;
      pancreas_*) recorded="${TH}/data/pancreas_thinning_050/corrupted.h5ad" ;;
      norman_*) recorded="${TT}/data/${key}/corrupted.h5ad" ;;
    esac
  elif (( i < 18 )); then
    group=deployment
    key=$(echo adamson_crispri papalexi_eccite norman_crispra pancreas_0 pancreas_1 pancreas_2 colon | cut -d' ' -f$((i - 10)))
    local d
    case "${key}" in
      adamson_crispri|papalexi_eccite|colon) d="${DEP}/${key}" ;;
      norman_crispra) d="${NKD}" ;;
      pancreas_*) d="${DEP}/pancreas/fold_${key##*_}" ;;
    esac
    [[ "${key}" == colon ]] && fit_split=validation
    input="${d}/hybrid.h5ad"; coordinates="${d}/coordinates.parquet"; splits="${d}/splits.parquet"
    recorded="${d}/recorded.h5ad"; layer=corrupted_counts; current="${d}/methods"
  elif (( i == 18 )); then
    group=protein; key=papalexi_crossmodal
    input="${PAPALEXI}/corrupted.h5ad"; coordinates="${PAPALEXI}/coordinates.parquet"; splits="${PAPALEXI}/splits.parquet"
    recorded=external_data/prepared/papalexi_eccite_crossmodal.h5ad; current="${PAPALEXI}"
  elif (( i < 22 )); then
    group=masked; key="colon_$((i - 19))"; fit_split=validation
    local d="${CCF}/fold_$((i - 19))"
    input="${d}/corrupted.h5ad"; coordinates="${d}/coordinates.parquet"; splits="${d}/splits.parquet"; recorded="${d}/prepared.h5ad"
  else
    group=thinned; fit_split=validation; layer=corrupted_counts
    key=$(echo colon_thinning_050_0 colon_thinning_050_1 colon_thinning_050_2 colon_thinning_025_0 colon_thinning_025_1 colon_thinning_025_2 | cut -d' ' -f$((i - 21)))
    local m="${CCF}/thinning/mask_trained/${key}/input"
    input="${m}/hybrid.h5ad"; coordinates="${m}/coordinates.parquet"; splits="${m}/splits.parquet"
    recorded="${CCF}/thinning/data/${key}/corrupted.h5ad"
  fi
  chain="${TEST}/${group}/${slug}/chains/${key}"
  new_input="${chain}/input/input.h5ad"
  new_coordinates="${chain}/input/coordinates.parquet"
  new_splits="${chain}/input/splits.parquet"
}

case "${stage}" in
  submit)
    mkdir -p logs
    inputs=$(sbatch --parsable -J v2t-inputs --array=0-18 --cpus-per-task=2 --mem=24G --time=00:30:00 "${SCRIPT}" inputs)
    cpu=$(sbatch --parsable -J v2t-cpu --array=0-75 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" cpu)
    gpu=$(sbatch --parsable -J v2t-gpu --array=0-18 --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" gpu)
    selector=$(sbatch --parsable -J v2t-sel --array=0-18 -p cs --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" selector)
    current=$(sbatch --parsable -J v2t-current --array=0-7 -p cs "${SCRIPT}" current_selector)
    tcpu=$(sbatch --parsable -J v2t-tcpu --array=0-14 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" trans_cpu)
    tmagic=$(sbatch --parsable -J v2t-tmagic --array=0-4 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" trans_magic)
    tscvi=$(sbatch --parsable -J v2t-tscvi --array=0-4 --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" trans_scvi)
    tsel=$(sbatch --parsable -J v2t-tsel --array=0-4 -p cs --dependency=afterok:${tcpu}:${tmagic}:${tscvi} "${SCRIPT}" trans_selector)
    evaluate=$(sbatch --parsable -J v2t-eval --cpus-per-task=8 --mem=64G --time=03:00:00 \
      --dependency=afterok:${selector}:${current}:${tsel} "${SCRIPT}" evaluate)
    echo "inputs=${inputs} cpu=${cpu} gpu=${gpu} selector=${selector} current=${current} trans_cpu=${tcpu}" \
         "trans_magic=${tmagic} trans_scvi=${tscvi} trans_selector=${tsel} evaluate=${evaluate}"
    ;;
  submit_colon_crossfit)
    mkdir -p logs
    inputs=$(sbatch --parsable -J v2t-inputs --array=19-27 --cpus-per-task=2 --mem=24G --time=00:30:00 "${SCRIPT}" inputs)
    cpu=$(sbatch --parsable -J v2t-cpu --array=76-111 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" cpu)
    gpu=$(sbatch --parsable -J v2t-gpu --array=19-27 --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" gpu)
    selector=$(sbatch --parsable -J v2t-sel --array=19-27 -p cs --dependency=afterok:${cpu}:${gpu} "${SCRIPT}" selector)
    tcpu=$(sbatch --parsable -J v2t-tcpu --array=15-23 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" trans_cpu)
    tmagic=$(sbatch --parsable -J v2t-tmagic --array=5-7 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" trans_magic)
    tscvi=$(sbatch --parsable -J v2t-tscvi --array=5-7 --gres=gpu:l40s:1 --mem=32G --dependency=afterok:${inputs} "${SCRIPT}" trans_scvi)
    tsel=$(sbatch --parsable -J v2t-tsel --array=5-7 -p cs --dependency=afterok:${tcpu}:${tmagic}:${tscvi} "${SCRIPT}" trans_selector)
    echo "inputs=${inputs} cpu=${cpu} gpu=${gpu} selector=${selector} trans_cpu=${tcpu} trans_magic=${tmagic}" \
         "trans_scvi=${tscvi} trans_selector=${tsel}; submit the evaluate stage after every chain has finished"
    ;;
  inputs)
    chain_paths "${task}"
    "${PY}" scripts/v2_selector_inputs.py refit --design "${design}" --rho "${rho}" --seed 1729 \
      --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --recorded "${recorded}" --recorded-layer "${layer}" --output-dir "${chain}/input"
    ;;
  cpu)
    chain_paths $((task / 4))
    method="${CPU_TEACHERS[task % 4]}"
    if [[ "${method}" == magic_inductive ]]; then
      .conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic \
        --input "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
        --output "${chain}/teachers/${method}" --seed 1729 --n-jobs "${OMP_NUM_THREADS}"
    else
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" \
        --input "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
        --output "${chain}/teachers/${method}" --seed 1729
    fi
    ;;
  gpu)
    chain_paths "${task}"
    .conda-scvi-current/bin/python scripts/v2_selector_scvi_teacher.py \
      --input "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
      --output "${chain}/teachers/scvi_inductive" --seed 1729
    ;;
  selector)
    chain_paths "${task}"
    "${PY}" scripts/v2_selector_fit.py --input "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
      --teachers-root "${chain}/teachers" --fit-split "${fit_split}" --features "${features}" \
      --output-dir "${TEST}/${group}/${slug}/scores/${key}" --seed 1729
    ;;
  current_selector)
    chain_paths $((task + 11))
    "${PY}" scripts/v2_selector_fit.py --input "${input}" --coordinates "${coordinates}" --splits "${splits}" \
      --teachers-root "${current}" --fit-split "${fit_split}" --features base \
      --output-dir "${TEST}/${group}/current/scores/${key}" --seed 1729
    ;;
  trans_cpu)
    chain_paths "${TRANS_CHAINS[task / 3]}"
    method="${TRANS_CPU[task % 3]}"
    "${PY}" scripts/run_leakage_safe_method.py --method "${method}" --transductive \
      --input "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
      --output "${chain}/teachers_transductive/${method}" --seed 1729
    ;;
  trans_magic)
    chain_paths "${TRANS_CHAINS[task]}"
    .conda-magic-current/bin/python scripts/run_magic_baseline.py \
      --corrupted "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
      --output "${chain}/standard/magic" --n-jobs "${OMP_NUM_THREADS}" --seed 1729
    "${PY}" scripts/count_scale_contract.py --contract "${chain}/standard/magic" --corrupted "${new_input}" \
      --output "${chain}/teachers_transductive/magic"
    ;;
  trans_scvi)
    chain_paths "${TRANS_CHAINS[task]}"
    .conda-scvi-current/bin/python scripts/run_scvi_baseline.py \
      --corrupted "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
      --output "${chain}/standard/scvi" --epochs 200 --seed 1729
    "${PY}" scripts/count_scale_contract.py --contract "${chain}/standard/scvi" --corrupted "${new_input}" \
      --output "${chain}/teachers_transductive/scvi"
    ;;
  trans_selector)
    chain_paths "${TRANS_CHAINS[task]}"
    "${PY}" scripts/v2_selector_fit.py --input "${new_input}" --coordinates "${new_coordinates}" --splits "${new_splits}" \
      --teachers-root "${chain}/teachers_transductive" --teacher-names gene_median svd_impute graph_smooth magic scvi \
      --fit-split "${fit_split}" --features "${features}" \
      --output-dir "${TEST}/masked/${slug}_transductive/scores/${key}" --seed 1729
    ;;
  evaluate)
    "${PY}" scripts/v2_selector_test_evaluate.py \
      --unit-source "${LF}/units_manifest.json" artifacts/paper_evidence/review_round2/fusion_value \
      --unit-source "${CCF}/units_manifest.json" "${CCF}/fusion_value" \
      --unit-keys pancreas_0 pancreas_1 pancreas_2 colon_0 colon_1 colon_2 norman_crispra \
      --v2-scores "Safe Fusion v2" "${TEST}/masked/${slug}/scores" \
      --v2-scores "Safe Fusion v2 (transductive)" "${TEST}/masked/${slug}_transductive/scores" \
      --references "Safe Fusion v2" "Safe Fusion v2 (transductive)" "Safe Fusion" "Safe Fusion (transductive)" \
      --output-dir "${TEST}/evaluation/masked"
    "${PY}" scripts/v2_selector_thinned_evaluate.py --colon-root "${CCF}" --v2-scores "Safe Fusion v2" "${TEST}/thinned/${slug}/scores" \
      --output-dir "${TEST}/evaluation/thinned"
    "${PY}" scripts/v2_selector_references.py --analysis knockdown sex \
      --scores current "${TEST}/deployment/current/scores" --scores v2 "${TEST}/deployment/${slug}/scores" \
      --reference current --output-dir "${TEST}/evaluation/references"
    "${PY}" scripts/v2_selector_references.py --analysis protein \
      --scores current "${TEST}/protein/current/scores" --scores v2 "${TEST}/protein/${slug}/scores" \
      --reference current --output-dir "${TEST}/evaluation/references"
    ;;
  *)
    echo "unknown stage ${stage}" >&2
    exit 2
    ;;
esac
echo "elapsed_s=${SECONDS}"
