#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
out=artifacts/paper_evidence/review_round4/table2_added
code=scripts/analyses/table2_added
[[ ! -e "$out/jobs.tsv" ]] || { echo 'Existing submission; inspect its jobs before starting a new run' >&2; exit 1; }
submit() {
    local stage=$1; shift
    local id
    id=$(sbatch --parsable -A torch_pr_634_general -J "t2add-$stage" -o "$out/logs/$stage-%A_%a.log" "$@")
    id=${id%%;*}
    printf '%s\t%s\n' "$id" "$stage" >> "$out/jobs.tsv"
    printf '%s' "$id"
}
audit=$(submit audit -p cs -c 2 --mem=64G -t 00:30:00 "$code/run.sh" "$code/audit_inputs.py")
recover=$(submit screcover -p cs --dependency="afterok:$audit" --array=0-7 -c 32 --mem=96G -t 18:00:00 "$code/run.sh" "$code/fit.py" screcover)
gpt=$(submit scgpt -p h200_courant --dependency="afterok:$audit" --array=0-7 -c 8 --mem=64G --gres=gpu:h200:1 -t 02:00:00 "$code/run.sh" "$code/fit.py" scgpt)
reference=$(submit reference -p cs --dependency="afterok:$audit" -c 2 --mem=8G -t 01:00:00 "$code/run.sh" "$code/evaluate.py" reference)
eval=$(submit evaluate -p cs --dependency="afterok:$recover:$gpt:$reference" -c 2 --mem=8G -t 01:00:00 "$code/run.sh" "$code/evaluate.py" added)
report=$(submit report -p cs --dependency="afterok:$eval" -c 2 --mem=8G -t 00:10:00 "$code/run.sh" "$code/report.py")
printf 'audit=%s screcover=%s scgpt=%s reference=%s evaluate=%s report=%s\n' "$audit" "$recover" "$gpt" "$reference" "$eval" "$report"
