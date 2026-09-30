#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
O=artifacts/paper_evidence/review_round4/sle_transductive
python "scripts/analyses/sle_transductive/setup.py"
submit() { sbatch --parsable -A torch_pr_634_general -p cs --output="$O/logs/%x-%A_%a.out" --error="$O/logs/%x-%A_%a.err" "$@"; }
vf=$(submit -J sf-r4-sle-vfills --array=0-11 --export=ALL,STAGE=verify_fills "scripts/analyses/sle_transductive/pipeline.sh")
ve=$(submit -J sf-r4-sle-verify --array=0-6 --mem=128G --dependency=afterok:$vf --export=ALL,STAGE=evaluate,VARIANT=verification "scripts/analyses/sle_transductive/pipeline.sh")
vc=$(submit -J sf-r4-sle-check --dependency=afterok:$ve "scripts/analyses/sle_transductive/validate.sh")
c=$(submit -J sf-r4-sle-chain --array=0-11 --export=ALL,STAGE=chain "scripts/analyses/sle_transductive/pipeline.sh")
e1=$(submit -J sf-r4-sle-cond --array=0-6 --mem=128G --dependency=afterok:$vc:$c --export=ALL,STAGE=evaluate,VARIANT=conditional "scripts/analyses/sle_transductive/pipeline.sh")
e2=$(submit -J sf-r4-sle-det --array=0-6 --mem=128G --dependency=afterok:$vc:$c --export=ALL,STAGE=evaluate,VARIANT=conditional_detection "scripts/analyses/sle_transductive/pipeline.sh")
audit=$(submit -J sf-r4-sle-audit --dependency=afterok:$c --cpus-per-task=2 --mem=16G --time=01:00:00 --wrap=".venv-sle/bin/python scripts/analyses/sle_transductive/audit.py")
ap=$(submit -J sf-r4-sle-ap --dependency=afterok:$c --cpus-per-task=8 --mem=32G --time=02:00:00 --wrap="export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8; .venv-sle/bin/python scripts/analyses/sle_transductive/ap_bootstrap.py")
submit -J sf-r4-sle-report --dependency=afterok:$e1:$e2:$audit:$ap --cpus-per-task=2 --mem=8G --time=00:30:00 --wrap=".venv-sle/bin/python scripts/analyses/sle_transductive/check_comparators.py && .venv-sle/bin/python scripts/analyses/sle_transductive/report.py"
printf 'verification fills=%s evaluations=%s check=%s chain=%s conditional=%s detection=%s\n' "$vf" "$ve" "$vc" "$c" "$e1" "$e2"
