#!/bin/bash
set -euo pipefail
O=artifacts/paper_evidence/review_round4/protein_multi/corrected
P=.venv/bin/python
(cd "$O" && sha256sum -c selection/FROZEN.sha256)
I=$O/prepared/deploy/hybrid.h5ad
C=$O/prepared/deploy/coordinates.parquet
S=$O/prepared/deploy/splits.parquet
common=(--coordinates "$C" --splits "$S" --seed 1729)
teachers=()
for name in gene_median svd_impute graph_smooth magic scvi; do teachers+=(--teacher-contract "$O/teachers/$name"); done
case "$STAGE" in
 cpu)
  for method in gene_median svd_impute graph_smooth; do
   $P scripts/run_leakage_safe_method.py --method "$method" --transductive --input "$I" "${common[@]}" --output "$O/teachers/$method"
  done ;;
 magic)
  .conda-magic-current/bin/python scripts/run_magic_baseline.py --corrupted "$I" "${common[@]}" --n-jobs "$SLURM_CPUS_PER_TASK" --output "$O/baselines/magic"
  $P scripts/count_scale_contract.py --contract "$O/baselines/magic" --corrupted "$I" --output "$O/teachers/magic" ;;
 scvi)
  .conda-scvi-current/bin/python scripts/run_scvi_baseline.py --corrupted "$I" "${common[@]}" --epochs 200 --output "$O/baselines/scvi"
  $P scripts/count_scale_contract.py --contract "$O/baselines/scvi" --corrupted "$I" --output "$O/teachers/scvi"
  .conda-scvi-current/bin/python scripts/nonzero_probability.py --method scvi --contract "$O/baselines/scvi" --corrupted "$I" --output "$O/baselines/scvi_probability" ;;
 fusion)
  $P scripts/run_leakage_safe_method.py --method safe_fusion --transductive --input "$I" "${common[@]}" "${teachers[@]}" --output "$O/safe_fusion" ;;
 selector)
  mapfile -t genes < <($P -c 'import pandas as pd,sys; print("\n".join(pd.read_csv(sys.argv[1]).gene.unique()))' "$O/selection/selected_pairs.csv")
  $P "scripts/analyses/protein_multi/selector.py" --positive-overflow-balanced --corrupted "$I" --truth "$O/prepared/truth.h5ad" "${common[@]}" --fusion-contract "$O/safe_fusion" "${teachers[@]}" --output-dir "$O/selector" --fit-split development --architecture mlp --budget-mode apply_topk --budgets 0.02 --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 --score-gene "${genes[@]}" ;;
esac
