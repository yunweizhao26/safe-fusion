#!/usr/bin/env bash
#SBATCH --job-name=sf-detection-rule
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=0-9
#SBATCH --output=logs/slurm-detection-rule-%A_%a.out
#SBATCH --error=logs/slurm-detection-rule-%A_%a.err

# Fill-fraction rule. The selector is refitted with the production settings;
# its scores are calibrated by isotonic regression on cross-fitted scores of
# the fitting cells, and every held-out zero whose detection probability
# p / (p + rho (1 - p)) exceeds 1/2 is filled, with rho = 0.10, the masking rate
# of every stratum. No held-out label or held-out score quantile sets the fill
# fraction.
#
# Tasks 0-4 apply the rule to the masked benchmark; the top-k output at 5% is a
# reproducibility check against the production selector. Tasks 5-9 apply it to
# the deployment input of the same units (scripts/deployment_paths.sh), in
# which the fitting cells keep the benchmark mask and the held-out cells hold
# their recorded counts, with the teachers and the stacked value refitted on
# that input by slurm_deployment_{prepare,scvi,fill}.sh. There every held-out
# candidate is a recorded zero, and detection_rule.test_fill_fraction is the
# share of those recorded zeros that the rule fills.
set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
source scripts/unit_paths.sh
source scripts/deployment_paths.sh

keys=(colon pancreas_0 pancreas_1 pancreas_2 norman_crispra)
key="${keys[SLURM_ARRAY_TASK_ID % ${#keys[@]}]}"
deployment_paths "${key}"
if (( SLURM_ARRAY_TASK_ID < ${#keys[@]} )); then
  data=(--corrupted "${input}" --coordinates "${coordinates}" --splits "${splits}")
  models="${methods_root}"
  output="artifacts/paper_evidence/detection_rule/${key}"
else
  data=(--corrupted "${out}/hybrid.h5ad" --coordinates "${out}/coordinates.parquet" --splits "${out}/splits.parquet")
  models="${deploy_methods}"
  output="${out}/detection_rule"
fi
mapfile -t teacher_args < <(teacher_contract_args "${models}")

.venv/bin/python scripts/calibrated_selective_fill.py \
  "${data[@]}" --truth "${truth}" \
  --fusion-contract "${models}/safe_fusion" "${teacher_args[@]}" \
  --output-dir "${output}" "${fit[@]}" \
  --architecture mlp --budget-mode apply_topk --budgets 0.05 \
  --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
  --detection-rule-mask-rate 0.10 --seed 1729 > /dev/null
