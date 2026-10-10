#!/bin/bash
#SBATCH -A torch_pr_634_general
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --constraint=l40s
#SBATCH --comment="preemption=yes;requeue=true"
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export STAGE=scvi
bash scripts/analyses/protein_multi/job.sh
