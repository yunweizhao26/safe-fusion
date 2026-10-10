#!/usr/bin/env bash
set -euo pipefail
export SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
BASE="$SAFE_FUSION_ROOT/artifacts/paper_evidence/review_round4/protein_full"
OUT="$BASE/full"
SCRIPTS="$SAFE_FUSION_ROOT/scripts"
CODE="$SCRIPTS/analyses/protein_full"
FOLLOWUP="$SAFE_FUSION_ROOT/artifacts/paper_evidence/review_round4/protein_followup"
PY="$SAFE_FUSION_ROOT/.venv/bin/python"
PREPARED="$OUT/prepared.h5ad"
PANEL="$SAFE_FUSION_ROOT/artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet"
ORIGINAL="$SAFE_FUSION_ROOT/external_data/prepared/papalexi_eccite_crossmodal.h5ad"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export XDG_CACHE_HOME="$BASE/cache" MPLCONFIGDIR="$BASE/cache/matplotlib" TORCH_HOME="$BASE/cache/torch"
export NUMBA_CACHE_DIR="$BASE/cache/numba" TMPDIR="$BASE/tmp/$SLURM_JOB_ID"
mkdir -p "$OUT" "$TMPDIR" "$XDG_CACHE_HOME"
cd "$SAFE_FUSION_ROOT"
echo "START $1 $(date -Is) job=$SLURM_JOB_ID node=$(hostname)"
common=(--input "$OUT/corrupted.h5ad" --coordinates "$OUT/coordinates.parquet" --splits "$OUT/splits.parquet" --seed 1729)
teachers=(--teacher-contract "$OUT/teachers/gene_median" --teacher-contract "$OUT/teachers/svd_impute" --teacher-contract "$OUT/teachers/graph_smooth" --teacher-contract "$OUT/teachers/magic" --teacher-contract "$OUT/teachers/scvi_counts")
case "$1" in
 prepare)
  "$PY" "$CODE/prepare.py" --input "$SAFE_FUSION_ROOT/external_data/scperturb/PapalexiSatija2021_eccite_RNA.h5ad" --output "$PREPARED" --dataset papalexi_eccite --matched-panel "$PANEL" --gene-panel "$ORIGINAL" --seed 1729
  "$PY" "$SCRIPTS/build_papalexi_crossmodal_benchmark.py" --input "$PREPARED" --output-dir "$OUT" --seed 1729
  ;;
 teachers)
  for method in gene_median svd_impute graph_smooth; do
   "$PY" "$SCRIPTS/run_leakage_safe_method.py" --method "$method" --transductive "${common[@]}" --output "$OUT/teachers/$method"
  done
  ;;
 baselines)
  for method in svd_impute graph_smooth; do
   "$PY" "$SCRIPTS/run_leakage_safe_method.py" --method "$method" "${common[@]}" --output "$OUT/$method"
  done
  ;;
 magic)
  "$SAFE_FUSION_ROOT/.conda-magic-current/bin/python" "$SCRIPTS/run_magic_baseline.py" --corrupted "$OUT/corrupted.h5ad" --coordinates "$OUT/coordinates.parquet" --splits "$OUT/splits.parquet" --output "$OUT/magic_standard" --n-jobs "$SLURM_CPUS_PER_TASK" --seed 1729
  "$PY" "$SCRIPTS/count_scale_contract.py" --contract "$OUT/magic_standard" --corrupted "$OUT/corrupted.h5ad" --output "$OUT/teachers/magic"
  ;;
 scvi)
  "$SAFE_FUSION_ROOT/.conda-scvi-current/bin/python" "$SCRIPTS/run_scvi_baseline.py" --corrupted "$OUT/corrupted.h5ad" --coordinates "$OUT/coordinates.parquet" --splits "$OUT/splits.parquet" --output "$OUT/scvi" --epochs 200 --seed 1729
  "$PY" "$SCRIPTS/count_scale_contract.py" --contract "$OUT/scvi" --corrupted "$OUT/corrupted.h5ad" --output "$OUT/teachers/scvi_counts"
  ;;
 stack)
  "$PY" "$SCRIPTS/run_leakage_safe_method.py" --method safe_fusion --transductive "${common[@]}" --output "$OUT/safe_fusion" "${teachers[@]}"
  ;;
 selector)
  mkdir -p "$FOLLOWUP/full"
  "$PY" "$CODE/selector.py" --selftest
  mapfile -t genes < <("$PY" -c 'import sys,anndata; print("\n".join(anndata.read_h5ad(sys.argv[1],backed="r").var_names))' "$OUT/corrupted.h5ad")
  "$PY" "$CODE/selector.py" --scale-balanced --corrupted "$OUT/corrupted.h5ad" --truth "$PREPARED" \
    --coordinates "$OUT/coordinates.parquet" --splits "$OUT/splits.parquet" \
    --fusion-contract "$OUT/safe_fusion" "${teachers[@]}" \
    --fit-split development --architecture mlp --budget-mode apply_topk \
    --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --seed 1729 \
    --output-dir "$FOLLOWUP/full/selector" --budgets 0.01 0.05 0.1 --score-gene "${genes[@]}"
  ;;
 evaluate)
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  "$PY" "$CODE/evaluate_full.py"
  ;;
 *) echo "Unknown stage $1" >&2; exit 2 ;;
esac
echo "END $1 $(date -Is)"
