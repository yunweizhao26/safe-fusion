#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/sle_donor_labels
export PYTHONDONTWRITEBYTECODE=1
python "scripts/analyses/sle_donor_labels/setup.py"
submit() {
  local jid
  jid=$(sbatch --parsable -A torch_pr_634_general -p cs --output="$O/logs/%x-%A_%a.out" --error="$O/logs/%x-%A_%a.err" "$@") || return $?
  printf '%s\t%s\n' "$jid" "$*" >> "$O/runtime/jobs.tsv"
  echo "$jid"
}
vf=$(submit -J sdl-replay-fills --array=0-11 --export=ALL,STAGE=verify_fills "scripts/analyses/sle_donor_labels/pipeline.sh")
v1=$(submit -J sdl-replay-fused --array=0-6 --mem=128G --dependency=afterok:$vf --export=ALL,STAGE=evaluate,VARIANT=verification "scripts/analyses/sle_donor_labels/pipeline.sh")
v2=$(submit -J sdl-replay-detect --array=0-6 --mem=128G --dependency=afterok:$vf --export=ALL,STAGE=evaluate,VARIANT=verification_detection "scripts/analyses/sle_donor_labels/pipeline.sh")
check=$(submit -J sdl-replay-check --dependency=afterok:$v1:$v2 --cpus-per-task=2 --mem=8G --time=00:30:00 --wrap=".venv-sle/bin/python scripts/analyses/sle_donor_labels/validate.py")
gpu=$(submit -J sdl-scvi --partition=l40s_public --gres=gpu:l40s:1 --time=01:00:00 --mem=48G --array=0-11 --dependency=afterok:$check --export=ALL,STAGE=scvi "scripts/analyses/sle_donor_labels/pipeline.sh")
chain=$(submit -J sdl-chain --array=0-11 --dependency=aftercorr:$gpu --export=ALL,STAGE=chain "scripts/analyses/sle_donor_labels/pipeline.sh")
e1=$(submit -J sdl-fused --array=0-6 --mem=128G --dependency=afterok:$chain --export=ALL,STAGE=evaluate,VARIANT=conditional "scripts/analyses/sle_donor_labels/pipeline.sh")
e2=$(submit -J sdl-detect --array=0-6 --mem=128G --dependency=afterok:$chain --export=ALL,STAGE=evaluate,VARIANT=conditional_detection "scripts/analyses/sle_donor_labels/pipeline.sh")
audit=$(submit -J sdl-audit --dependency=afterok:$chain --cpus-per-task=2 --mem=16G --time=01:00:00 --wrap=".venv-sle/bin/python scripts/analyses/sle_donor_labels/audit.py")
ap=$(submit -J sdl-ap --dependency=afterok:$chain --cpus-per-task=8 --mem=32G --time=02:00:00 --wrap="export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8; .venv-sle/bin/python scripts/analyses/sle_donor_labels/ap_bootstrap.py")
submit -J sdl-report --dependency=afterok:$e1:$e2:$audit:$ap --cpus-per-task=2 --mem=8G --time=00:30:00 --wrap=".venv-sle/bin/python scripts/analyses/sle_donor_labels/check_comparators.py && .venv-sle/bin/python scripts/analyses/sle_donor_labels/report.py"
