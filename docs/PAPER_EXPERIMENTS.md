# Paper experiment index

This index maps each experiment family in the manuscript to repository code.
Commands, inputs, output contracts, and result paths are recorded in
[REPRODUCTION.md](REPRODUCTION.md) and
[PAPER_EVIDENCE.md](PAPER_EVIDENCE.md).

| Paper experiment | Preparation and methods | Evaluation and figure code |
|---|---|---|
| Masked recovery and matched fill fractions | `workflow/Snakefile`, `scripts/run_leakage_safe_method.py`, `scripts/calibrated_selective_fill.py`, `scripts/selector_attribution.py` | `scripts/locked_primary_analysis.py`, `scripts/compute_matched_baseline_f1_curves.py`, `scripts/combine_mlp_baseline_curves.py`, `scripts/plot_selector_f1_fillrate.py` |
| Selector and fusion attribution | `scripts/selector_attribution.py`, `scripts/slurm_selector_*.sh` | `scripts/combine_selector_attribution.py`, `scripts/summarize_selector_attribution.py` |
| ALRA, SAVER, MAGIC, scVI, scGCL, and scGPT comparisons | `scripts/run_*_baseline.py`, `scripts/run_saver_baseline.R`, `scripts/run_scgpt_gene_prediction.py` | `scripts/evaluate_baseline_contract.py`, `scripts/compute_matched_baseline_f1_curves.py` |
| Pancreas and colon markers, ectopic filling, and donor effects | `scripts/prepare_pancreas_atlas.py`, `scripts/prepare_colon_atlas.py` | `scripts/evaluate_pancreas_crossfit_biology.py`, `scripts/evaluate_colon_donor_biology.py`, `scripts/evaluate_curated_biology.py`, `scripts/evaluate_postfacto_markers.py` |
| Cell-type clustering and differential expression | `scripts/slurm_prepare_complete_downstream_teachers.sh`, `scripts/slurm_complete_downstream_selectors.sh` | `scripts/slurm_evaluate_complete_downstream.sh`, `scripts/summarize_complete_downstream.py`, `scripts/plot_complete_downstream.py` |
| Zebrafish trajectory preservation | `scripts/prepare_zebrafish_trajectory.py`, `scripts/run_leakage_safe_method.py` | `scripts/evaluate_trajectory_preservation.py` |
| Simulated and perturbation-based GRN sensitivity | `scripts/grn_sensitivity_sergio.py`, `scripts/prepare_external_perturbseq.py` | `scripts/evaluate_interventional_grn.py` |
| Norman CRISPRa target and signature preservation | `scripts/prepare_norman_crispra.py`, `scripts/setup_norman_crispra_experiment.py` | `scripts/evaluate_perturbation_preservation.py` |
| Adamson CRISPRi, Dixit KO, and Papalexi ECCITE-seq | `scripts/prepare_external_perturbseq.py`, `scripts/build_papalexi_crossmodal_benchmark.py` | `scripts/evaluate_papalexi_crossmodal.py`, `scripts/evaluate_perturbation_preservation.py` |
| PBMC protein and PD-L1 validation | `scripts/prepare_pbmc_citeseq.py`, `scripts/calibrated_selective_fill.py` | `scripts/evaluate_protein_biology.py`, `scripts/plot_biological_range_figures.py` |
| Crohn disagreement analysis (Figure 1) | Source counts, imputed matrices, and concordance JSON supplied with `--input-root` to `scripts/regenerate_legacy_disagreement_plots.py` | same script; set `--output-dir` to your result directory |

Generated matrices, contracts, logs, and local baseline checkouts remain
outside Git. Each run writes hashes and provenance beside its outputs. Use
MLP `apply_topk` for the manuscript method and logistic selection for its
corresponding ablation.
