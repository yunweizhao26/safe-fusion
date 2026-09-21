#!/bin/bash
#SBATCH --job-name=sf_scgpt
#SBATCH --output=logs/sf_scgpt_%j.log
#SBATCH --error=logs/sf_scgpt_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Frozen scGPT whole-human embeddings for the locked datasets. Requires the
# baseline env and the checkpoint from scripts/sbatch_setup_python_baselines.s.

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

CKPT=external_data/baselines/scgpt_human
OUT=artifacts/paper_evidence/scgpt_embeddings
mkdir -p "$OUT"
.venv-baselines/bin/pip install -q "IPython"
.venv-baselines/bin/pip install -q --force-reinstall "torch==2.3.1+cu118" --index-url https://download.pytorch.org/whl/cu118

.venv-baselines/bin/python -u scripts/scgpt_embedding_runner.py \
  --input external_data/prepared/colon_epithelial.h5ad \
  --ckpt-dir "$CKPT" --output "$OUT/colon_epithelial.npy" --device cuda

.venv-baselines/bin/python -u scripts/scgpt_embedding_runner.py \
  --input external_data/prepared/pancreas_islets.h5ad \
  --ckpt-dir "$CKPT" --output "$OUT/pancreas_islets.npy" --device cuda

.venv-baselines/bin/python -u scripts/scgpt_embedding_runner.py \
  --input external_data/prepared/norman_crispra.h5ad \
  --ckpt-dir "$CKPT" --output "$OUT/norman_crispra.npy" --device cuda

echo SCGPT_EMBEDDINGS_DONE
