#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
root="$PWD/artifacts/paper_evidence/review_round4/scale_caps"
[[ ! -e "$root/jobs.tsv" || "${RECOVERY:-0}" == 1 ]] || { echo 'jobs.tsv exists; refusing duplicate submission' >&2; exit 2; }
if [[ ! -e "$root/jobs.tsv" ]]; then printf 'genes\tvariant\tstage\tjob\n' > "$root/jobs.tsv"; fi
submit() {
    local genes="$1" variant="$2" stage="$3" memory="$4" hours="$5" dep="$6" job
    local opts=()
    [[ -z "$dep" ]] || opts+=(--dependency="afterok:$dep" --kill-on-invalid-dep=yes)
    job=$(sbatch --parsable -A torch_pr_634_general -p cs --cpus-per-task=8 --mem="${memory}G" --time="${hours}:00:00" \
        --job-name="caps-${genes}-${variant}-${stage}" --output="$root/logs/%x-%j.out" --error="$root/logs/%x-%j.err" \
        "${opts[@]}" "scripts/analyses/scale_caps/run.sh" "$stage" "$genes" "$variant")
    printf '%s\t%s\t%s\t%s\n' "$genes" "$variant" "$stage" "$job" >> "$root/jobs.tsv"
    printf '%s' "$job"
}
gates=()
for genes in 5000 2000; do
    value=$(submit "$genes" default value 128 4 '')
    selector=$(submit "$genes" default selector 96 2 "$value")
    gates+=("$(submit "$genes" default check 64 1 "$selector")")
done
for genes in 5000 2000; do
    value=$(submit "$genes" large value 160 8 "${gates[0]}:${gates[1]}")
    selector=$(submit "$genes" large selector 128 6 "$value")
    submit "$genes" large evaluate 128 4 "$selector" >/dev/null
done
cat "$root/jobs.tsv"
