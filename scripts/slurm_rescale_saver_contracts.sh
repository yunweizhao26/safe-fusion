#!/usr/bin/env bash
#SBATCH --job-name=sf-saver-rescale
#SBATCH --account=torch_pr_634_general
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-saver-rescale-%j.out
#SBATCH --error=logs/slurm-saver-rescale-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

B=artifacts/paper_evidence/baselines/saver
.venv/bin/python scripts/rescale_saver_contracts.py --contract "${B}/pancreas_mask_010" \
  --corrupted artifacts/pancreas_runs/0b2469810675-45c81b160d78/data/pancreas_islets/corrupted/mask_010.h5ad
.venv/bin/python scripts/rescale_saver_contracts.py --contract "${B}/colon_mask_010" \
  --corrupted artifacts/colon_runs/0b2469810675-c0db6f963e94/data/colon_epithelial/corrupted/mask_010.h5ad
.venv/bin/python scripts/rescale_saver_contracts.py --contract "${B}/norman_full_mask_010" \
  --corrupted artifacts/paper_evidence/norman_crispra/corrupted.h5ad
