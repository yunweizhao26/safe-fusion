#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
C=scripts/analyses/transductive_downstream
python "$C/build_manifest.py"
cpu() { sbatch --wait -A torch_pr_634_general -p cs "$@"; }
for units in 0-23 24-29; do
  cpu --array="$units" "$C/job.sh" cpu
  sbatch --wait -A torch_pr_634_general -p l40s_public --gres=gpu:l40s:1 --mem=48G --array="$units" "$C/job.sh" gpu
  cpu --array="$units" "$C/job.sh" select
done
for part in full masked_only zeros_only; do
  cpu --array=0-7 --mem=64G "$C/job.sh" evaluate "$part"
done
cpu --array=8-23 --mem=64G "$C/job.sh" evaluate full
for analysis in effects modules annotation; do
  cpu --array=8-10,12-14 "$C/analysis.sh" disease "$analysis"
done
cpu --array=0-2,8-10,16-18 "$C/analysis.sh" disease_endpoints
for design in deployment null; do
  cpu "$C/analysis.sh" correlation "$design"
done
cpu --array=0-3 "$C/reproduce.sh"
cpu "$C/analysis.sh" summarize
cpu "$C/analysis.sh" audit
cpu "$C/analysis.sh" report
