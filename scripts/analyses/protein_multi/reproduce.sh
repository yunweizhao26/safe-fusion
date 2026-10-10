#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
O=artifacts/paper_evidence/review_round4/protein_multi/corrected
(cd "$O" && sha256sum -c selection/FROZEN.sha256)
sha256sum -c "$O/runtime/source_sha256.txt"
bash "scripts/analyses/protein_multi/submit_fits.sh"
selector=$(sed -n 's/^selector=//p' "$O/runtime/fit_jobs.txt")
bash "scripts/analyses/protein_multi/submit_evaluation.sh" "$selector"
while squeue -u "$USER" -h -o '%j' | grep -q '^pm4-'; do sleep 600; done
