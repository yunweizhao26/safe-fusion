#!/usr/bin/env bash
#SBATCH --job-name=sf-combine-mlp-attr
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/slurm-combine-mlp-attribution-range-%j.out
#SBATCH --error=logs/slurm-combine-mlp-attribution-range-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

.venv/bin/python scripts/combine_selector_attribution.py \
  --input-dir artifacts/paper_evidence/selector_mlp_attribution_range/pancreas/fold_0 \
  --input-dir artifacts/paper_evidence/selector_mlp_attribution_range/pancreas/fold_1 \
  --input-dir artifacts/paper_evidence/selector_mlp_attribution_range/pancreas/fold_2 \
  --output-dir artifacts/paper_evidence/selector_mlp_attribution_range/pancreas/combined \
  --bootstrap 2000 --seed 1729
