#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE=artifacts/paper_evidence/review_round4/protein_full
mkdir -p "$BASE" logs
[[ ! -e "$BASE/jobs.tsv" ]] || { echo 'jobs.tsv exists; refusing duplicate submission' >&2; exit 2; }
submit() {
 local stage=$1 cpus=$2 mem=$3 time=$4 dep=$5
 local dependency=() resource=(-p cs) job
 [[ "$dep" == none ]] || dependency=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
 [[ "$stage" != scvi ]] || resource=(--gres=gpu:l40s:1)
 job=$(env -u SBATCH_PARTITION sbatch --parsable -A torch_pr_634_general "${resource[@]}" \
  -J "protein-full-$stage" -c "$cpus" --mem="$mem" -t "$time" \
  --output="logs/protein-full-$stage-%j.out" --error="logs/protein-full-$stage-%j.err" \
  "${dependency[@]}" scripts/analyses/protein_full/stage.sh "$stage") || return $?
 printf '%s\t%s\n' "$stage" "$job" >> "$BASE/jobs.tsv"
 echo "$job"
}
prepare=$(submit prepare 4 48G 01:00:00 none)
teachers=$(submit teachers 8 64G 02:00:00 "$prepare")
baselines=$(submit baselines 8 64G 02:00:00 "$prepare")
magic=$(submit magic 8 64G 02:00:00 "$prepare")
scvi=$(submit scvi 8 48G 03:00:00 "$prepare")
stack=$(submit stack 8 64G 06:00:00 "$teachers:$magic:$scvi")
selector=$(submit selector 8 64G 04:00:00 "$stack")
submit evaluate 3 48G 03:00:00 "$selector:$baselines" >/dev/null
cat "$BASE/jobs.tsv"
