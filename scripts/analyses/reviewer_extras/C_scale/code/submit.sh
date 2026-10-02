#!/usr/bin/env bash

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
root="$PWD/artifacts/paper_evidence/review_round4/reviewer_extras/C_scale"
[[ ! -s "$root/jobs.tsv" ]] || { echo 'jobs.tsv already exists; inspect before resubmitting' >&2; exit 2; }
printf 'stage\tstep\tcells\tjob\n' > "$root/jobs.tsv"
submit() {
  local stage="$1" task="$2" memory="$3" hours="$4" dependency="$5" label="$6" job
  local options=(--partition=cs)
  case "$stage" in saver) options=(--partition=cl);; scvi|scvi_teacher) options=(--partition=l40s_public --gres=gpu:l40s:1);; esac
  [[ -z "$dependency" ]] || options+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
  job=$(sbatch --parsable -A torch_pr_634_general --cpus-per-task=8 --mem="${memory}G" --time="${hours}:00:00" \
    --job-name="extras-C-$label" --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" \
    "${options[@]}" "scripts/analyses/reviewer_extras/C_scale/code/run.sh" "$stage" "$task") || return $?
  printf '%s\t%s\t200000\t%s\n' "$stage" "$label" "$job" >> "$root/jobs.tsv"
  printf '%s' "$job"
}
prepare_dependency=''
[[ -z "${AFTER_B:-}" ]] || prepare_dependency="afterok:$AFTER_B"
prepare=$(submit prepare 0 128 2 "$prepare_dependency" prepare)
mask=$(submit mask 0 96 2 "afterok:$prepare" mask)
median=$(submit teachers 0 128 4 "afterok:$mask" gene_median)
svd=$(submit teachers 1 128 4 "afterok:$mask" svd)
knn=$(submit teachers 2 128 4 "afterok:$mask" knn)
magic_teacher=$(submit teachers 3 192 12 "afterok:$mask" magic_teacher)
scvi_teacher=$(submit scvi_teacher 0 128 10 "afterok:$mask" scvi_teacher)
base="afterok:$median:$svd:$knn:$scvi_teacher"
stack=$(submit stack 0 160 4 "$base:$magic_teacher" stack)
selector=$(submit selector 0 160 4 "afterok:$stack" selector)
without_stack=$(submit without_magic_stack 0 160 4 "$base" without_magic_stack)
without_selector=$(submit without_magic_selector 0 160 4 "afterok:$without_stack" without_magic_selector)
magic=$(submit magic 0 256 12 "afterok:$mask" magic)
scvi=$(submit scvi 0 128 8 "afterok:$mask" scvi)
probability=$(submit scvi_probability 0 128 2 "afterok:$scvi" scvi_probability)
alra=$(submit alra 0 128 4 "afterok:$mask" alra)
saver=$(submit saver 0 480 16 "afterok:$mask" saver)
tmedian=$(submit transductive_teachers 0 128 4 "afterok:$mask" transductive_median)
tsvd=$(submit transductive_teachers 1 128 4 "afterok:$mask" transductive_svd)
tknn=$(submit transductive_teachers 2 128 4 "afterok:$mask" transductive_knn)
tmagic=$(submit transductive_teachers 3 128 2 "afterok:$magic" transductive_magic)
tscvi=$(submit transductive_teachers 4 128 2 "afterok:$scvi" transductive_scvi)
tstack=$(submit transductive_stack 0 160 4 "afterok:$tmedian:$tsvd:$tknn:$tmagic:$tscvi" transductive_stack)
tselector=$(submit transductive_selector 0 160 4 "afterok:$tstack" transductive_selector)
evaluate=$(submit evaluate 0 160 6 "afterany:$selector:$without_selector:$tselector:$alra:$saver:$probability:$magic_teacher" evaluate)
cat "$root/jobs.tsv"
