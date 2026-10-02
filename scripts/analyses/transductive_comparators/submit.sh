#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/transductive_comparators
test ! -e "$O/jobs.tsv"
mkdir -p "$O/logs"
parity=$(sbatch --parsable -A torch_pr_634_general -p cs --array=0-2 --job-name=tcmp-parity -o "$O/logs/parity-%A_%a.log" "scripts/analyses/transductive_comparators/run.sh" parity)
alra=$(sbatch --parsable -A torch_pr_634_general -p cs --array=0-6 --dependency=afterok:"$parity" --kill-on-invalid-dep=yes --job-name=tcmp-alra -o "$O/logs/alra-%A_%a.log" "scripts/analyses/transductive_comparators/run.sh" alra)
evaluate=$(sbatch --parsable -A torch_pr_634_general -p cs --array=0-2 --dependency=afterok:"$alra" --kill-on-invalid-dep=yes --job-name=tcmp-eval -o "$O/logs/evaluate-%A_%a.log" "scripts/analyses/transductive_comparators/run.sh" evaluate)
report=$(sbatch --parsable -A torch_pr_634_general -p cs --dependency=afterok:"$evaluate" --kill-on-invalid-dep=yes --job-name=tcmp-report -o "$O/logs/report-%j.log" "scripts/analyses/transductive_comparators/run.sh" report)
printf 'stage\tjob_id\nparity\t%s\nalra\t%s\nevaluate\t%s\nreport\t%s\n' "$parity" "$alra" "$evaluate" "$report" > "$O/jobs.tsv"
