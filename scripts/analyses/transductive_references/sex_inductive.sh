#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
units=(pancreas_0 pancreas_1 pancreas_2 colon_0 colon_1 colon_2)
u=${units[SLURM_ARRAY_TASK_ID]}
R=artifacts/paper_evidence/review_round4/transductive_references/sex_zeros
D=$R/deployment_rebuilt/$u
O=$R/inductive_rebuilt/$u
common=(--input "$D/hybrid.h5ad" --coordinates "$D/coordinates.parquet" --splits "$D/splits.parquet" --seed 1729)
case "$STAGE" in
teachers)
for m in gene_median svd_impute graph_smooth; do .venv/bin/python scripts/run_leakage_safe_method.py --method "$m" "${common[@]}" --output "$O/$m"; done
.conda-magic-current/bin/python scripts/run_inductive_teacher.py --method magic "${common[@]}" --output "$O/magic_inductive" --n-jobs "$SLURM_CPUS_PER_TASK"
;;
scvi)
.conda-scvi-current/bin/python scripts/run_inductive_teacher.py --method scvi "${common[@]}" --output "$O/scvi_inductive"
;;
select)
teachers=()
for m in gene_median svd_impute graph_smooth magic_inductive scvi_inductive; do teachers+=(--teacher-contract "$O/$m"); done
.venv/bin/python scripts/run_leakage_safe_method.py --method safe_fusion "${common[@]}" --output "$O/safe_fusion" "${teachers[@]}"
if [[ $u == pancreas_* ]]; then S=artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit/fold_${u##*_}; fit=development; else S=artifacts/paper_evidence/review_round3/colon_crossfit/fold_${u##*_}; fit=validation; fi
.venv/bin/python scripts/calibrated_selective_fill.py --corrupted "$D/hybrid.h5ad" --truth "$S/prepared.h5ad" --coordinates "$D/coordinates.parquet" --splits "$D/splits.parquet" --fusion-contract "$O/safe_fusion" "${teachers[@]}" --output-dir "$O/selector" --fit-split "$fit" --architecture mlp --budget-mode apply_topk --budgets 0.01 0.05 0.10 --curve-points 2 --score-gene ENSG00000229807 ENSG00000129824 --seed 1729
;;
esac
