#!/bin/bash
set -euo pipefail
f=$SLURM_ARRAY_TASK_ID
S=artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit/fold_$f
.venv/bin/python scripts/build_deployment_inputs.py --truth "$S/prepared.h5ad" --corrupted "$S/corrupted.h5ad" --coordinates "$S/coordinates.parquet" --splits "$S/splits.parquet" --output-dir "artifacts/paper_evidence/review_round4/transductive_references/sex_zeros/deployment_rebuilt/pancreas_$f"
