#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/transductive_comparators
while squeue -u yz5944 -h -n tcmp-parity,tcmp-alra,tcmp-eval,tcmp-report | grep -q .; do
    date -u
    squeue -u yz5944 -h -n tcmp-parity,tcmp-alra,tcmp-eval,tcmp-report -o '%i %j %T %M %R'
    sleep 600
done
ids=$(awk 'NR>1 {print $2}' "$O/jobs.tsv" | paste -sd, -)
sacct -j "$ids" --units=M --parsable2 --format=JobID,JobName,Partition,Account,State,ExitCode,Elapsed,TotalCPU,AllocCPUS,ReqMem,MaxRSS,NodeList > "$O/jobs_final.tsv"
cat "$O/jobs_final.tsv"
