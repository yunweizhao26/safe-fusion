#!/usr/bin/env bash

set -euo pipefail
SOURCE=artifacts/paper_evidence/review_round4/sex_zero_comparators
export SEX_ZERO_OUTPUT=${SEX_ZERO_OUTPUT:-${SOURCE}/reproduction_run}
if [[ -e ${SEX_ZERO_OUTPUT} ]]; then echo "Use a fresh SEX_ZERO_OUTPUT directory" >&2; exit 1; fi
mkdir -p "${SEX_ZERO_OUTPUT}"/{logs,fits,kcluster,verification,evaluation}
common=(-A torch_pr_634_general --parsable --output="${SEX_ZERO_OUTPUT}/logs/%x-%A_%a.log")
fits=()
for method in alra saver dca kcluster; do
  jid=$(sbatch "${common[@]}" --array=0-5 --job-name=sf-sz-${method} --export=ALL,METHOD=${method} "scripts/analyses/sex_zero_comparators/fit.sh")
  echo "$method $jid" >> "${SEX_ZERO_OUTPUT}/jobs.txt"
  if [[ $method == kcluster ]]; then cluster=$jid; else fits+=("$jid"); fi
done
for method in scimpute enimpute; do
  jid=$(sbatch "${common[@]}" --array=0-5 --dependency=afterok:${cluster} --job-name=sf-sz-${method} --export=ALL,METHOD=${method} "scripts/analyses/sex_zero_comparators/fit.sh")
  echo "$method $jid" >> "${SEX_ZERO_OUTPUT}/jobs.txt"
  fits+=("$jid")
done
verify=$(sbatch "${common[@]}" --array=0-1 --job-name=sf-sz-verify --export=ALL,STAGE=verify "scripts/analyses/sex_zero_comparators/evaluate.sh")
dependency=$(IFS=:; echo "${fits[*]}")
evaluate=$(sbatch "${common[@]}" --array=0-1 --dependency=afterany:${dependency},afterok:${verify} --job-name=sf-sz-evaluate --export=ALL,STAGE=comparators "scripts/analyses/sex_zero_comparators/evaluate.sh")
echo "verify $verify" >> "${SEX_ZERO_OUTPUT}/jobs.txt"
echo "evaluate $evaluate" >> "${SEX_ZERO_OUTPUT}/jobs.txt"
sbatch "${common[@]}" -p cs -c 2 --mem=8G --time=00:10:00 --dependency=afterok:${evaluate} --job-name=sf-sz-report --wrap=".venv/bin/python scripts/analyses/sex_zero_comparators/report.py"
