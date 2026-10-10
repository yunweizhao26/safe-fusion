#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
O=artifacts/paper_evidence/review_round4/label_baselines/screens
V=artifacts/paper_evidence/review_round4/transductive_references/knockdown/evaluation_inputs
.venv/bin/python scripts/evaluate_knockdown_zero_analyses.py --deploy-root "$V/deployment" --review-root "$V/review" --norman-root "$V/norman" --norman-benchmark artifacts/paper_evidence/review_round2/leakage_free/norman_crispra --output-dir "$O/verification_transductive" --depth-strata 5 --draws 2000 --seed 1729
.venv/bin/python scripts/standard_imputers/evaluate_r3_known_zeros.py --output-dir "$O/verification_inductive_comparators" --parts knockdown --depth-strata 5 --draws 2000 --seed 1729
.venv/bin/python "scripts/analyses/label_baselines/screens/verify_tables.py"
