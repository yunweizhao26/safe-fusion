#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
while squeue -u yz5944 -h -o '%j' | grep -q '^r2x-'; do sleep 600; done
O=artifacts/paper_evidence/review_round4/round2_extras
ids=$(cut -f1 "$O/reproduce_jobs.tsv" | paste -sd,)
sacct -j "$ids" -X -P --format=JobID,JobName,State,Elapsed,ExitCode,AllocCPUS,ReqMem > "$O/job_accounting.txt"
sacct -j "$ids" -P --units=M --format=JobID,JobName,State,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,ExitCode > "$O/resource_accounting.txt"
