#!/usr/bin/env bash
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="$PWD/artifacts/paper_evidence/review_round4/scale_comparators"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8
export XDG_CACHE_HOME="$OUT/cache" MPLCONFIGDIR="$OUT/cache/matplotlib" NUMBA_CACHE_DIR="$OUT/cache/numba"
export KERAS_HOME="$OUT/cache/keras" CUDA_CACHE_PATH="$OUT/cache/cuda"
export TMPDIR="$OUT/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$OUT/cache"
stage=$1
size=$2
unit="$PWD/artifacts/paper_evidence/review_round3/scale/cells_$size"
result="$OUT/cells_$size"
common=(--corrupted "$unit/masked/corrupted.h5ad" --coordinates "$unit/masked/coordinates.parquet" --splits "$unit/splits.parquet")
case "$stage" in
  report)
    .venv/bin/python -u "scripts/analyses/scale_comparators/report.py"
    .venv/bin/python -u "scripts/analyses/scale_comparators/check_results.py"
    ;;
  reproduce) .venv/bin/python -u "scripts/analyses/scale_comparators/evaluate.py" --size "$size" --reproduce ;;
  evaluate|evaluate_ready) .venv/bin/python -u "scripts/analyses/scale_comparators/evaluate.py" --size "$size" ;;
  evaluate_gpu) .venv/bin/python -u "scripts/analyses/scale_comparators/evaluate.py" --size "$size" --core-only --include-scgpt ;;
  evaluate_core) .venv/bin/python -u "scripts/analyses/scale_comparators/evaluate.py" --size "$size" --core-only ;;
  check_selector)
    .venv/bin/python -u scripts/stacked_selector_scores.py "${common[@]}" --fit-split development --unit-column donor --contract "$unit/baselines/scvi" --name scVI --output-dir "$OUT/checks/selector_scvi" --seed 1729
    .venv/bin/python -u "scripts/analyses/scale_comparators/check_selector.py"
    ;;
  kcluster) .venv-scanpy/bin/python -u scripts/standard_imputers/r3_kcluster.py --corrupted "$unit/masked/corrupted.h5ad" --output "$result/kcluster.json" --seed 1729 ;;
  selector_scvi|selector_magic)
    method=${stage#selector_}
    .venv/bin/python -u "scripts/analyses/scale_comparators/scale_selector.py" "${common[@]}" --fusion-contract "$unit/transductive/$method" --teacher-contract "$unit/transductive/$method" --output-dir "$result/$stage" --stacked-convention --seed 1729 ;;
  dca|scimpute|screcover|enimpute)
    .venv/bin/python -u scripts/standard_imputers/run_r3_comparator.py "${common[@]}" --method "$stage" --output "$result/fits/$stage" --kcluster-json "$result/kcluster.json" --ncores 8 --seed 1729 ;;
  scgpt)
    .venv-baselines/bin/python -u scripts/run_scgpt_gene_prediction.py --corrupted "$unit/masked/corrupted.h5ad" --coordinates "$unit/masked/coordinates.parquet" --ckpt-dir external_data/baselines/scgpt_human --output "$result/fits/scgpt" --batch-size 64 --gene-batch-size 512 --device cuda --seed 1729 ;;
  *) echo "Unknown stage $stage" >&2; exit 2 ;;
esac
