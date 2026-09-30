#!/bin/bash

set -euo pipefail
O=artifacts/paper_evidence/review_round4/transductive_references
S=scripts/analyses/transductive_references
cpu() { sbatch --wait -A torch_pr_634_general -p cs -c 8 --mem=48G -t 06:00:00 -o "$O/logs/reproduce-%j.log" -e "$O/logs/reproduce-%j.err" "$@"; }
gpu() { sbatch --wait -A torch_pr_634_general -p l40s_public --gres=gpu:l40s:1 -c 8 --mem=48G -t 03:00:00 -o "$O/logs/reproduce-%j.log" -e "$O/logs/reproduce-%j.err" "$@"; }

mkdir -p "$O/logs"
python "$S/setup_resume.py"
for analysis in knockdown protein sex; do cpu "$S/verify_inductive.sh" "$analysis"; done
cpu "$S/verify_results.sh"

for stage in teachers magic; do cpu --array=0-3 --export=ALL,STAGE=$stage scripts/slurm_transductive_knockdown_masked.sh; done
for stage in scvi scvi_condition; do gpu --array=0-3 --export=ALL,STAGE=$stage scripts/slurm_transductive_knockdown_masked.sh; done
for stage in stack select; do cpu --array=0-3 --export=ALL,STAGE=$stage scripts/slurm_transductive_knockdown_masked.sh; done
for stage in teachers reuse; do cpu --array=0-2 --export=ALL,STAGE=$stage scripts/slurm_transductive_knockdown_deploy.sh; done
gpu --array=0-2 --export=ALL,STAGE=scvi_condition scripts/slurm_transductive_knockdown_deploy.sh
for stage in stack select finalize; do cpu --array=0-2 --export=ALL,STAGE=$stage scripts/slurm_transductive_knockdown_deploy.sh; done
cpu "$S/knockdown_evaluate.sh"

cpu --array=0-2 "$S/sex_prepare.sh"
cpu --array=0-2 scripts/slurm_transductive_sex_zeros_colon_prepare.sh
for stage in teachers magic; do cpu --array=0-5 --export=ALL,STAGE=$stage "$S/sex_deploy.sh"; done
for stage in scvi scvi_donor; do gpu --array=0-5 --export=ALL,STAGE=$stage "$S/sex_deploy.sh"; done
for stage in stack select; do cpu --array=0-5 --export=ALL,STAGE=$stage "$S/sex_deploy.sh"; done
cpu --array=0-5 --export=ALL,STAGE=teachers "$S/sex_inductive.sh"
gpu --array=0-5 --export=ALL,STAGE=scvi "$S/sex_inductive.sh"
cpu --array=0-5 --export=ALL,STAGE=select "$S/sex_inductive.sh"
gpu --array=0-5 --export=ALL,STAGE=donor_teachers "$S/sex_masked.sh"
for stage in stack select; do cpu --array=0-5 --export=ALL,STAGE=$stage "$S/sex_masked.sh"; done
cpu "$S/sex_evaluate.sh" recorded_transductive
cpu "$S/sex_inductive_evaluate.sh"
cpu "$S/sex_evaluate.sh" masked

for stage in teachers magic stack; do cpu --export=ALL,STAGE=$stage scripts/slurm_transductive_protein.sh; done
python "$S/setup_resume.py"
for stage in select evaluate plot; do cpu --export=ALL,STAGE=$stage "$S/protein_cs.sh"; done
cpu --wrap=".venv/bin/python $S/report.py"
