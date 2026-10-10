#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ROOT=artifacts/paper_evidence/review_round4/reviewer_extras
DEP=()
[[ -n ${AFTER:-} ]] && DEP=(--dependency="afterok:${AFTER}")
check=$(sbatch --parsable -J extras-B-check -c 2 --mem=8G -t 00:20:00 "${DEP[@]}" -o "$ROOT/logs/B-check-%j.log" "scripts/analyses/reviewer_extras/job.sh" .venv/bin/python -u "scripts/analyses/reviewer_extras/B_thinning/run.py" check)
fit=$(sbatch --parsable -J extras-B-fit --array=0-7 --mem=64G -t 04:00:00 --dependency="afterok:$check" -o "$ROOT/logs/B-fit-%A_%a.log" "scripts/analyses/reviewer_extras/job.sh" bash "scripts/analyses/reviewer_extras/B_thinning/fit.sh")
summary=$(sbatch --parsable -J extras-B-summary -c 2 --mem=8G -t 00:20:00 --dependency="afterok:$fit" -o "$ROOT/logs/B-summary-%j.log" "scripts/analyses/reviewer_extras/job.sh" .venv/bin/python -u "scripts/analyses/reviewer_extras/B_thinning/run.py" summary)
printf 'stage\tjob\ncheck\t%s\nfit\t%s\nsummary\t%s\n' "$check" "$fit" "$summary" > "$ROOT/B_thinning/jobs.tsv"
cat "$ROOT/B_thinning/jobs.tsv"
