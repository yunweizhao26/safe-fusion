#!/bin/bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
S=scripts/analyses/transductive_references
.venv/bin/python "$S/sex_evaluate.py" recorded_inductive
.venv/bin/python "$S/combine_sex.py"
