#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
out=artifacts/paper_evidence/review_round4/fill_grid
[[ ! -e "$out/jobs.tsv" ]] || { echo 'Existing submission; do not overwrite.' >&2; exit 1; }
gate_units=$(sbatch --parsable -J fg-gate-unit --array=0-7 -o "$out/logs/gate-%A_%a.log" "scripts/analyses/fill_grid/job.sh" gate-unit)
printf 'stage\tjob\ngate-unit\t%s\n' "$gate_units" > "$out/jobs.tsv"
gate=$(sbatch --parsable -J fg-gate --dependency="afterok:$gate_units" -o "$out/logs/gate-summary-%j.log" "scripts/analyses/fill_grid/job.sh" gate)
printf 'gate\t%s\n' "$gate" >> "$out/jobs.tsv"
grid=$(sbatch --parsable -J fg-grid --array=0-7 --dependency="afterok:$gate" -o "$out/logs/grid-%A_%a.log" "scripts/analyses/fill_grid/job.sh" grid-unit)
printf 'grid-unit\t%s\n' "$grid" >> "$out/jobs.tsv"
summary=$(sbatch --parsable -J fg-summary --dependency="afterok:$grid" -o "$out/logs/summary-%j.log" "scripts/analyses/fill_grid/job.sh" summary)
printf 'summary\t%s\n' "$summary" >> "$out/jobs.tsv"
cat "$out/jobs.tsv"
