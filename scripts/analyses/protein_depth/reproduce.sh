#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
O=$PWD/artifacts/paper_evidence/review_round4/protein_depth
mkdir -p "$O/logs" "$O/runtime" "$O/inputs"
.venv/bin/python scripts/analyses/protein_depth/setup.py
submit() {
 sbatch --parsable -A torch_pr_634_general -p cs "$@"
}
preps=()
for kind in hao papalexi_small papalexi_full; do
 j=$(submit -J pd-prepare -c 8 --mem=96G -t 02:00:00 -o "$O/logs/reproduce-$kind-%j.out" -e "$O/logs/reproduce-$kind-%j.err" "scripts/analyses/protein_depth/stage.sh" prepare "$kind")
 preps+=("$j")
 printf '%s %s\n' "$kind" "$j" >> "$O/runtime/reproduce_jobs.txt"
done
dep=$(IFS=:; echo "${preps[*]}")
jmerge=$(submit -J pd-merge -c 1 --mem=4G -t 00:10:00 --dependency=afterok:$dep -o "$O/logs/reproduce-merge-%j.out" -e "$O/logs/reproduce-merge-%j.err" "scripts/analyses/protein_depth/stage.sh" merge)
jpairs=$(submit -J pd-pairs -c 1 --mem=8G -t 02:00:00 --array=0-42%12 --dependency=afterok:$jmerge -o "$O/logs/reproduce-pair-%A_%a.out" -e "$O/logs/reproduce-pair-%A_%a.err" "scripts/analyses/protein_depth/stage.sh" pair)
originals=()
for kind in papalexi_small papalexi_full; do
 j=$(submit -J pd-original -c 1 --mem=8G -t 02:00:00 --dependency=afterok:$jmerge -o "$O/logs/reproduce-original-$kind-%j.out" -e "$O/logs/reproduce-original-$kind-%j.err" "scripts/analyses/protein_depth/stage.sh" original "$kind")
 originals+=("$j")
done
all_dep=$(IFS=:; echo "$jpairs:${originals[*]}")
jreport=$(submit -J pd-report -c 1 --mem=8G -t 00:30:00 --dependency=afterok:$all_dep -o "$O/logs/reproduce-report-%j.out" -e "$O/logs/reproduce-report-%j.err" "scripts/analyses/protein_depth/stage.sh" report)
printf 'merge %s\npairs %s\noriginal %s\nreport %s\n' "$jmerge" "$jpairs" "${originals[*]}" "$jreport" >> "$O/runtime/reproduce_jobs.txt"
wait_ids=$(IFS=,; echo "${preps[*]},$jmerge,$jpairs,${originals[*]},$jreport")
while squeue -h -j "$wait_ids" | grep -q .; do sleep 600; done
sacct -j "$wait_ids" --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem,MaxRSS,TotalCPU --parsable2 > "$O/runtime/reproduce_accounting.psv"
