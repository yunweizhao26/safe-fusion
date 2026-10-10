#!/usr/bin/env bash
#SBATCH --job-name=sf-pap-xm-raw
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=logs/slurm-pap-xm-raw-%j.out
#SBATCH --error=logs/slurm-pap-xm-raw-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
ROOT=artifacts/paper_evidence/papalexi_crossmodal/benchmark
.venv/bin/python scripts/evaluate_papalexi_crossmodal.py \
  --prepared external_data/prepared/papalexi_eccite_crossmodal.h5ad \
  --corrupted "${ROOT}/corrupted.h5ad" \
  --panel artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet \
  --selector-scores "${ROOT}/mlp_selector/selected_gene_scores.parquet" \
  --fusion-mean "${ROOT}/safe_fusion/mean.npy" \
  --graph-mean "${ROOT}/graph_smooth/mean.npy" \
  --svd-mean "${ROOT}/svd_impute/mean.npy" \
  --scvi-mean "${ROOT}/scvi/mean.npy" \
  --output-dir "${ROOT}/evaluation_raw_counts" \
  --protein-column adt_count \
  --bootstrap 1000 \
  --permutations 1000 \
  --seed 1729
