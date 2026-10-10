#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH -p cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
O=$PWD/artifacts/paper_evidence/review_round4/protein_multi/corrected
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export XDG_CACHE_HOME=$O/cache MPLCONFIGDIR=$O/cache/matplotlib NUMBA_CACHE_DIR=$O/cache/numba
export TMPDIR=$O/cache/tmp
mkdir -p "$TMPDIR"
start=$(date +%s)
trap 'code=$?; printf "%s,%s,%s,%s\n" "$SLURM_JOB_ID" "$STAGE" "$(( $(date +%s)-start ))" "$code" > "$O/runtime/${SLURM_JOB_ID}.csv"' EXIT
case $STAGE in
 data) .venv/bin/python "scripts/analyses/protein_multi/prepare_data.py" ;;
 select)
  .venv/bin/python "scripts/analyses/protein_multi/select_pairs.py"
  .venv/bin/python scripts/make_mask_replicate.py --truth "$O/prepared/truth.h5ad" --splits "$O/prepared/splits.parquet" --unit-column donor --seed 1729 --output-dir "$O/prepared/masked"
  .venv/bin/python scripts/build_deployment_inputs.py --truth "$O/prepared/truth.h5ad" --corrupted "$O/prepared/masked/corrupted.h5ad" --coordinates "$O/prepared/masked/coordinates.parquet" --splits "$O/prepared/splits.parquet" --output-dir "$O/prepared/deploy"
  .venv/bin/python scripts/analyses/protein_multi/audit.py ;;
 cpu|magic|scvi|fusion|selector) bash "scripts/analyses/protein_multi/fit.sh" ;;
 extract)
  .venv/bin/python "scripts/analyses/protein_multi/evaluate.py" selftest
  .venv/bin/python "scripts/analyses/protein_multi/evaluate.py" extract ;;
 evaluate) .venv/bin/python "scripts/analyses/protein_multi/evaluate.py" pair ;;
 report)
  .venv/bin/python "scripts/analyses/protein_multi/evaluate.py" aggregate
  .venv/bin/python "scripts/analyses/protein_multi/final_audit.py"
  .venv/bin/python "scripts/analyses/protein_multi/report.py" ;;
 *) echo "Unknown stage $STAGE" >&2; exit 2 ;;
esac
