#!/usr/bin/env bash

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
root="$PWD/artifacts/paper_evidence/review_round4/reviewer_extras/C_scale"
[[ ! -e "$root/magic_recovery_complete" ]] || exit 2
submit() {
  local stage="$1" task="$2" dependency="$3" label="$4" job
  local dep=()
  [[ -z "$dependency" ]] || dep=(--dependency="$dependency" --kill-on-invalid-dep=yes)
  job=$(sbatch --parsable -A torch_pr_634_general -p cs -c 8 --mem=160G -t 06:00:00 \
    -J "extras-C-$label" -o "$root/logs/%x-%j.out" -e "$root/logs/%x-%j.err" \
    "${dep[@]}" "scripts/analyses/reviewer_extras/C_scale/code/run.sh" "$stage" "$task") || return $?
  printf '%s\t%s\t200000\t%s\n' "$stage" "$label" "$job" >> "$root/jobs.tsv"
  printf '%s\t%s\n' "$label" "$job" >> "$root/magic_recovery_jobs.tsv"
  printf '%s' "$job"
}
magic=$(awk -F '\t' '$1=="transductive_magic_recovery" {print $2}' "$root/magic_recovery_jobs.tsv" 2>/dev/null || true)
[[ -n "$magic" ]] || magic=$(submit transductive_teachers 3 '' transductive_magic_recovery)
stack=$(submit transductive_stack 0 "afterok:18885080:$magic" transductive_stack_recovery)
selector=$(submit transductive_selector 0 "afterok:$stack" transductive_selector_recovery)
scancel 18885225
evaluate=$(submit evaluate 0 "afterany:18885069:18885071:$selector:18885224:18885074:18885066" evaluate_recovery)
touch "$root/magic_recovery_complete"
cat "$root/magic_recovery_jobs.tsv"
