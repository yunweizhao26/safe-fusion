#!/bin/bash

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/label_baselines
test ! -e "$O/masked/replay"
s16=$(sbatch --parsable -A torch_pr_634_general -p cs -J lb-screen-verify -c 2 --mem=64G -t 04:00:00 -o "$O/screens/verify_%j.log" "scripts/analyses/label_baselines/screens/verify.sh")
s18=$(sbatch --parsable --array=0-1 -J lb-sex-verify --export=ALL,STAGE=verify "scripts/analyses/label_baselines/sex/run.sh")

gate=$(sbatch --parsable -A torch_pr_634_general -p cs -J lb-gate-audited -c 1 --mem=1G -t 00:05:00 --dependency="afterany:$s16:$s18" -o "$O/gate-%j.log" --wrap="PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/analyses/label_baselines/verify_gate.py")
check=$(sbatch --parsable --dependency="afterok:$gate" "scripts/analyses/label_baselines/masked/selfcheck.sh")
replay=$(sbatch --parsable --dependency="afterok:$gate" "scripts/analyses/label_baselines/masked/replay.sh")
masked=$(sbatch --parsable --dependency="afterok:$check:$replay" "scripts/analyses/label_baselines/masked/run.sh")
masked_extra=$(sbatch --parsable --dependency="afterok:$masked" "scripts/analyses/label_baselines/masked/supplementary.sh")
screens=$(sbatch --parsable -A torch_pr_634_general -p cs -J lb-screen-b -c 2 --mem=64G -t 02:00:00 --dependency="afterok:$gate" -o "$O/screens/baselines_%j.log" "scripts/analyses/label_baselines/screens/baselines.sh")
sex=$(sbatch --parsable --array=0-1 -J lb-sex-analysis --export=ALL,STAGE=analyze --dependency="afterok:$gate" "scripts/analyses/label_baselines/sex/run.sh")
sex_extra=$(sbatch --parsable --array=0-1 -J lb-sex-transductive --export=ALL,STAGE=transductive --dependency="afterok:$sex" "scripts/analyses/label_baselines/sex/run.sh")
jobs="$s16,$s18,$gate,$check,$replay,$masked,$masked_extra,$screens,$sex,$sex_extra"
printf '%s\n' "$jobs" > "$O/reproduce_job_ids.txt"
while true; do
  queue=$(squeue -h -j "$jobs" -o '%i %T %r')
  test -n "$queue" || break
  if grep -q DependencyNeverSatisfied <<< "$queue"; then
    printf '%s\n' "$queue" >&2
    exit 1
  fi
  sleep 600
done
sacct -j "$jobs" --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,TotalCPU,MaxRSS -P > "$O/complete_job_accounting.txt"
sbatch -A torch_pr_634_general -p cs -J lb-final-report -c 1 --mem=2G -t 00:10:00 -o "$O/report-%j.log" --wrap="set -e; export PYTHONDONTWRITEBYTECODE=1; .venv/bin/python scripts/analyses/label_baselines/masked/summarize.py; .venv/bin/python scripts/analyses/label_baselines/sex/report.py; .venv/bin/python scripts/analyses/label_baselines/report.py"
