#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONDONTWRITEBYTECODE=1
O=artifacts/paper_evidence/review_round4/reviewer_extras
.venv/bin/python "scripts/analyses/reviewer_extras/C_scale/code/report.py" > "$O/logs/C-final-report-${SLURM_JOB_ID}.md"
.venv/bin/python "scripts/analyses/reviewer_extras/report.py"
cat "$O/README.md"
