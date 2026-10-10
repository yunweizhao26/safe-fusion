#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
out=artifacts/paper_evidence/review_round4/scale_comparators
export PYTHONDONTWRITEBYTECODE=1

deadline=$((SLURM_JOB_END_TIME - 7200))
while (( $(date +%s) < deadline )); do
  date -u
  .venv/bin/python "scripts/analyses/scale_comparators/advance.py"
  .venv/bin/python "scripts/analyses/scale_comparators/report.py"
  [[ -f "$out/DONE" || -f "$out/BLOCKED.json" ]] && break
  sleep 600
done
