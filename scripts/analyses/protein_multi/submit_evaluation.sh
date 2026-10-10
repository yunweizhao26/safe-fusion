#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
O=artifacts/paper_evidence/review_round4/protein_multi/corrected
mkdir -p "$O/logs" "$O/runtime"
submit() { sbatch --parsable -A torch_pr_634_general -p cs --output="$O/logs/%x-%A_%a.out" --error="$O/logs/%x-%A_%a.err" "$@" "scripts/analyses/protein_multi/job.sh"; }
extract=$(submit -J pm4-extract --cpus-per-task=2 --mem=24G --dependency=afterok:${1:?selector job ID required} --export=ALL,STAGE=extract)
n=$(($(wc -l < "$O/selection/selected_pairs.csv")-2))
evaluate=$(submit -J pm4-eval --cpus-per-task=1 --mem=4G --array=0-$n%8 --dependency=afterok:$extract --export=ALL,STAGE=evaluate)
report=$(submit -J pm4-report --cpus-per-task=2 --mem=8G --dependency=afterok:$evaluate --export=ALL,STAGE=report)
printf '%s\n' "extract=$extract" "evaluate=$evaluate" "report=$report" | tee "$O/runtime/evaluation_jobs.txt"
