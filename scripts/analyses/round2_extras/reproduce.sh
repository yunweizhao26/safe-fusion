#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/round2_extras
part=${1:-all}
case "$part" in all|A|B|C) ;; *) echo 'Use all, A, B or C' >&2; exit 2;; esac
submit() {
  local stage=$1; shift
  local id
  id=$(sbatch --parsable -A torch_pr_634_general -p cs -J "r2x-$stage" -o "$O/logs/$stage-%A_%a.log" "$@")
  printf '%s\t%s\n' "$id" "$stage" >> "$O/reproduce_jobs.tsv"
  printf '%s' "$id"
}
if [[ $part == all || $part == A ]]; then
  g=$(submit gate -c 2 --mem=8G -t 00:10:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_a/analyze.py" gate)
  ids=()
  for endpoint in adamson_crispri norman_crispra pancreas colon; do
    ids+=("$(submit "a-$endpoint" --dependency="afterok:$g" -c 2 --mem=64G -t 02:00:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_a/analyze.py" "$endpoint")")
  done
  dep=$(IFS=:; echo "${ids[*]}")
  a=$(submit asum --dependency="afterok:$dep" -c 2 --mem=8G -t 00:10:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_a/summarize.py")
fi
if [[ $part == all || $part == B ]]; then
  b=$(submit b -c 8 --mem=64G -t 02:00:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_b/curves.py")
fi
if [[ $part == all || $part == C ]]; then
  c0=$(submit c0 --array=0-6 -c 2 --mem=48G -t 02:00:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_c/evaluate.py" 0)
  en=$(submit en16 --array=0-27 -c 16 --mem=96G -t 24:00:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_c/fit.py" enimpute)
  fast=$(submit fast16 --array=0-27 -c 16 --mem=64G -t 04:00:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_c/fit.py" fast)
  ce=$(submit ceval --dependency="afterany:$en:$fast" --array=0-27 -c 2 --mem=48G -t 02:00:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_c/evaluate.py" 7)
  cs=$(submit csum --dependency="afterany:$c0:$ce" -c 2 --mem=8G -t 00:20:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/part_c/evaluate.py" summary)
fi
if [[ $part == all ]]; then
  report=$(submit report --dependency="afterany:$a:$b:$cs" -c 2 --mem=8G -t 00:10:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/report.py")
  submit audit --dependency="afterok:$report" -c 2 --mem=8G -t 00:10:00 "scripts/analyses/round2_extras/run.sh" "scripts/analyses/round2_extras/audit.py"
fi
