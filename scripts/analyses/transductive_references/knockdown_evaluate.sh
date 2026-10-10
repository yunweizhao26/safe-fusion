#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
K=artifacts/paper_evidence/review_round4/transductive_references/knockdown
V=$K/evaluation_inputs
N=artifacts/paper_evidence/review_round2/leakage_free/norman_crispra
.venv/bin/python scripts/evaluate_knockdown_zero_analyses.py --deploy-root "$V/deployment" --review-root "$V/review" --norman-root "$V/norman" --norman-benchmark "$N" --output-dir "$K/evaluation_matched" --depth-strata 5 --draws 2000 --seed 1729
if [[ ! -f "$K/masked_f1/masked_f1_by_screen.json" ]]; then
.venv/bin/python scripts/evaluate_condition_masked_f1.py --external-root "$V/masked" --norman-benchmark "$N" --norman-methods "$N/methods" --norman-label-methods "$V/masked/norman_crispra" --norman-truth "$N/prepared.h5ad" --norman-selector "$K/masked/norman_crispra/selector_mlp_biology_range" --output-dir "$K/masked_f1" --unit-output-dir "$K/masked_f1" --draws 2000 --seed 1729
fi
