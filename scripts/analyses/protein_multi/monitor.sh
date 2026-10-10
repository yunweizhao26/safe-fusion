#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
O=artifacts/paper_evidence/review_round4/protein_multi/corrected
while squeue -u "$USER" -h -o '%j' | grep -q '^pm4-'; do
 sleep 600
 date -u > "$O/runtime/latest_queue.txt"
 squeue -u "$USER" -h -o '%i %j %T %M %R' | grep ' pm4-' >> "$O/runtime/latest_queue.txt" || true
 cat "$O/runtime/latest_queue.txt"
 if grep -q 'DependencyNeverSatisfied' "$O/runtime/latest_queue.txt"; then exit 1; fi
done
