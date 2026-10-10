#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/sle_donor_labels
touch "$O/runtime/superseded_job_ids.txt"
while squeue -u yz5944 -h -o '%j' | grep -q '^sdl-'; do
  sleep 600
  date -u
  squeue -u yz5944 -h -o '%i %j %T %M %R' | grep 'sdl-' || true
  ids=$(cut -f1 "$O/runtime/jobs.tsv" | paste -sd,)
  sacct -j "$ids" -X -n -o JobID,JobName,State,Elapsed,ExitCode > "$O/runtime/sacct_records.txt"
  awk 'FILENAME==ARGV[1] {skip[$1]=1; next} {split($1,a,"_"); if (!skip[a[1]]) print}' "$O/runtime/superseded_job_ids.txt" "$O/runtime/sacct_records.txt" > "$O/runtime/active_sacct_records.txt"
  if grep -Eq 'FAILED|OUT_OF_MEMORY|TIMEOUT|CANCELLED|NODE_FAIL' "$O/runtime/active_sacct_records.txt"; then
    cat "$O/runtime/sacct_records.txt"
    exit 1
  fi
done
