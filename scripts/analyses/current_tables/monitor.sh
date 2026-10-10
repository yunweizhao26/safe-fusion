#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/current_tables
while squeue -u yz5944 -h -o '%j' | grep -q '^ct-'; do
 sleep 600
 squeue -u yz5944 -h -o '%i %j %T %M %R' | grep ' ct-' || true
 if rg -l 'Traceback|AssertionError|ValueError|OUT_OF_MEMORY|TIME LIMIT' "$O/logs"; then break; fi
done
ids=$(cut -f1 "$O/jobs.tsv" | paste -sd,)
sacct -j "$ids" --format=JobID,JobName,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocCPUS -P > "$O/accounting.tsv"
cat "$O/accounting.tsv"
