#!/usr/bin/env bash
#SBATCH --job-name=sf-r4-protein
#SBATCH --account=torch_pr_634_general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-r4-protein-%j.out
#SBATCH --error=logs/slurm-r4-protein-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

SRC=artifacts/paper_evidence/papalexi_crossmodal/benchmark
OUT=artifacts/paper_evidence/review_round4/transductive_references/protein
PREPARED=external_data/prepared/papalexi_eccite_crossmodal.h5ad
PY=.venv/bin/python

common=(--input "${OUT}/corrupted.h5ad" --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" --seed 1729)

case "${STAGE:?STAGE must be teachers, magic, stack, select, evaluate or plot}" in
  teachers)
    for method in gene_median svd_impute graph_smooth; do
      "${PY}" scripts/run_leakage_safe_method.py --method "${method}" --transductive "${common[@]}" \
        --output "${OUT}/${method}"
    done
    ;;
  magic)
    .conda-magic-current/bin/python scripts/run_magic_baseline.py \
      --corrupted "${OUT}/corrupted.h5ad" --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
      --output "${OUT}/magic_standard" --n-jobs "${SLURM_CPUS_PER_TASK}" --seed 1729
    "${PY}" scripts/count_scale_contract.py --contract "${OUT}/magic_standard" --corrupted "${OUT}/corrupted.h5ad" \
      --output "${OUT}/magic"
    "${PY}" scripts/count_scale_contract.py --contract "${SRC}/scvi" --corrupted "${OUT}/corrupted.h5ad" \
      --output "${OUT}/scvi_counts"
    ;;
  stack)
    "${PY}" scripts/run_leakage_safe_method.py --method safe_fusion --transductive "${common[@]}" \
      --output "${OUT}/safe_fusion" \
      --teacher-contract "${OUT}/gene_median" --teacher-contract "${OUT}/svd_impute" \
      --teacher-contract "${OUT}/graph_smooth" --teacher-contract "${OUT}/magic" \
      --teacher-contract "${OUT}/scvi_counts"
    ;;
  select)
    teacher_args=(--teacher-contract "${OUT}/gene_median" --teacher-contract "${OUT}/svd_impute" \
      --teacher-contract "${OUT}/graph_smooth" --teacher-contract "${OUT}/magic" --teacher-contract "${OUT}/scvi_counts")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${OUT}/corrupted.h5ad" --truth "${PREPARED}" \
      --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
      --fusion-contract "${OUT}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${OUT}/mlp_selector" --fit-split development --architecture mlp \
      --budget-mode apply_topk --budgets 0.02 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
      --score-gene CD274 CD86 PDCD1LG2 HAVCR2 --seed 1729
    mapfile -t genes < <("${PY}" -c \
      'import sys, anndata; print("\n".join(anndata.read_h5ad(sys.argv[1], backed="r").var_names))' \
      "${OUT}/corrupted.h5ad")
    "${PY}" scripts/calibrated_selective_fill.py \
      --corrupted "${OUT}/corrupted.h5ad" --truth "${PREPARED}" \
      --coordinates "${OUT}/coordinates.parquet" --splits "${OUT}/splits.parquet" \
      --fusion-contract "${OUT}/safe_fusion" "${teacher_args[@]}" \
      --output-dir "${OUT}/global_fill_selector" --fit-split development --architecture mlp \
      --budget-mode apply_topk --budgets 0.01 0.05 0.1 \
      --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
      --score-gene "${genes[@]}" --seed 1729
    ;;
  evaluate)
    "${PY}" scripts/evaluate_papalexi_crossmodal.py \
      --prepared "${PREPARED}" --corrupted "${OUT}/corrupted.h5ad" \
      --panel "artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet" \
      --selector-scores "${OUT}/mlp_selector/selected_gene_scores.parquet" \
      --fusion-mean "${OUT}/safe_fusion/mean.npy" --graph-mean "${OUT}/graph_smooth/mean.npy" \
      --svd-mean "${OUT}/svd_impute/mean.npy" --scvi-mean "${SRC}/scvi/mean.npy" \
      --output-dir "${OUT}/evaluation" --bootstrap 1000 --permutations 1000 --seed 1729
    for column in adt_clr adt_count; do
      output="${OUT}/evaluation_cd274"
      [[ "${column}" == adt_count ]] && output="${OUT}/evaluation_cd274_raw_counts"
      "${PY}" scripts/evaluate_papalexi_crossmodal.py \
        --prepared "${PREPARED}" --corrupted "${OUT}/corrupted.h5ad" \
        --panel "artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet" \
        --selector-scores "${OUT}/mlp_selector/selected_gene_scores.parquet" \
        --fusion-mean "${OUT}/safe_fusion/mean.npy" --graph-mean "${OUT}/graph_smooth/mean.npy" \
        --svd-mean "${OUT}/svd_impute/mean.npy" --scvi-mean "${SRC}/scvi/mean.npy" \
        --output-dir "${output}" --protein-column "${column}" --gene CD274 \
        --bootstrap 1000 --permutations 1000 --seed 1729
    done
    "${PY}" scripts/evaluate_pdl1_state_baselines.py \
      --prepared "${PREPARED}" --benchmark-root "${OUT}" \
      --panel "artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet" \
      --output-dir "${OUT}/evaluation_pdl1_state" --bootstrap 1000 --seed 1729
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      "${PY}" scripts/evaluate_protein_within_state.py \
      --prepared "${PREPARED}" --benchmark-root "${OUT}" \
      --panel "artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet" \
      --global-fill-dir "${OUT}/global_fill_selector" --output-dir "${OUT}/evaluation_within_state" \
      --bootstrap 2000 --seed 1729 --workers "${SLURM_CPUS_PER_TASK}"
    ;;
  plot)
    MPLBACKEND=Agg MPLCONFIGDIR="/tmp/sf-r4-pdl1-${SLURM_JOB_ID}" \
      .venv-baselines/bin/python scripts/plot_biological_range_figures.py \
      --pdl1-only --crossmodal-root "${OUT}/evaluation_cd274" \
      --output-dir "${OUT}/figures" --paper-dir "${OUT}/figures"
    ;;
  *)
    echo "unknown STAGE ${STAGE}" >&2
    exit 2
    ;;
esac
