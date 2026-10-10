#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
O=artifacts/paper_evidence/review_round4/protein_multi/corrected
submit() { sbatch --parsable -A torch_pr_634_general --output="$O/logs/%x-%j.out" --error="$O/logs/%x-%j.err" "$@" "scripts/analyses/protein_multi/job.sh"; }
mkdir -p "$O/logs" "$O/runtime"
dep=()
if [[ -n ${1:-} ]]; then dep=(--dependency=afterok:$1); fi
cpu=$(submit -J pm4-cpu "${dep[@]}" --export=ALL,STAGE=cpu)
magic=$(submit -J pm4-magic "${dep[@]}" --export=ALL,STAGE=magic)
scvi=$(sbatch --parsable -A torch_pr_634_general -J pm4-scvi-flex --output="$O/logs/%x-%j.out" --error="$O/logs/%x-%j.err" "${dep[@]}" "scripts/analyses/protein_multi/gpu_job.sh")
fusion=$(submit -J pm4-fusion --dependency=afterok:$cpu:$magic:$scvi --export=ALL,STAGE=fusion)
selector=$(submit -J pm4-selector -p cs --mem=128G --dependency=afterok:$fusion --export=ALL,STAGE=selector)
printf '%s\n' "cpu=$cpu" "magic=$magic" "scvi=$scvi" "fusion=$fusion" "selector=$selector" | tee "$O/runtime/fit_jobs.txt"
