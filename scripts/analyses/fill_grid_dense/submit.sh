#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/fill_grid_dense
C=scripts/analyses/fill_grid_dense
[[ ! -e "$O/jobs.tsv" ]] || { echo 'Existing submission; do not overwrite.' >&2; exit 1; }
mkdir -p "$O/logs" "$O/tmp"
gate_units=$(sbatch --parsable -A torch_pr_634_general -p cs -J fgd-gate-unit --array=0-7%2 -o "$O/logs/gate-%A_%a.log" "$C/job.sh" gate-unit)
printf 'stage\tjob\ngate-unit\t%s\n' "$gate_units" > "$O/jobs.tsv"
gate=$(sbatch --parsable -A torch_pr_634_general -p cs -J fgd-gate --dependency="afterok:$gate_units" -o "$O/logs/gate-%j.log" "$C/job.sh" gate)
printf 'gate\t%s\n' "$gate" >> "$O/jobs.tsv"
dense=$(sbatch --parsable -A torch_pr_634_general -p cs -J fgd-dense-unit --array=0-7%2 --dependency="afterok:$gate" -o "$O/logs/dense-%A_%a.log" "$C/job.sh" dense-unit)
printf 'dense-unit\t%s\n' "$dense" >> "$O/jobs.tsv"
summary=$(sbatch --parsable -A torch_pr_634_general -p cs -J fgd-summary --dependency="afterok:$dense" -o "$O/logs/summary-%j.log" "$C/job.sh" summary)
printf 'summary\t%s\n' "$summary" >> "$O/jobs.tsv"
cat "$O/jobs.tsv"
