#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=artifacts/paper_evidence/review_round4/transductive_references/verification
case "$1" in
knockdown)
.venv/bin/python scripts/evaluate_knockdown_zero_analyses.py --output-dir "$O/knockdown" --depth-strata 5 --draws 2000 --seed 1729
N=artifacts/paper_evidence/review_round2/leakage_free/norman_crispra
.venv/bin/python scripts/evaluate_condition_masked_f1.py --norman-benchmark "$N" --norman-methods "$N/methods" --norman-label-methods artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt/masked --norman-truth "$N/prepared.h5ad" --norman-selector artifacts/paper_evidence/review_round2/leakage_free/selector_mlp_biology_range_fullteachers/norman_crispra --output-dir "$O/masked_f1" --unit-output-dir "$O/masked_f1" --draws 2000 --seed 1729
;;
protein)
.venv/bin/python scripts/evaluate_protein_within_state.py --output-dir "$O/protein" --bootstrap 2000 --seed 1729 --workers 8
;;
sex)
.venv/bin/python - "$O" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,'scripts')
import disease_control_sex_zeros as e
root=Path(sys.argv[1])
e.output_dir=lambda analysis,tissue: (root/analysis/tissue)
for tissue in ('pancreas','colon'):
 (root/'sex_zeros'/tissue).mkdir(parents=True,exist_ok=True)
 sys.argv=['disease_control_sex_zeros.py','--tissue',tissue,'--draws','2000','--seed','1729']
 e.main()
PY
;;
esac
