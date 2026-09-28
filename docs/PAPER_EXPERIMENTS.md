# Manuscript items, scripts and outputs

This table maps every table and figure of the manuscript to the launchers
and scripts that produce it and to its output files. Every launcher is in
`scripts/`, and every output path is under `artifacts/paper_evidence/` unless
it starts with `artifacts/` or `<methods_root>`. `<methods_root>` is the
contract directory of a dataset in `scripts/unit_paths.sh`.
[REPRODUCTION.md](REPRODUCTION.md) gives the commands and their dependencies,
and the section column refers to it. Every item also needs the data of
section 2 and the fits of section 3 (teachers:
`slurm_prepare_complete_downstream_teachers.sh`, `slurm_scvi_teachers.sh`,
`run_leakage_safe_method.py`, `run_inductive_teacher.py`; fused value:
`slurm_safe_fusion_stack.sh`; selector: `slurm_complete_downstream_selectors.sh`,
`calibrated_selective_fill.py`).

| Item | Content | Launchers | Scripts | Outputs | Section |
|---|---|---|---|---|---|
| Figure 1 | Masked F1 at 1000 fill fractions | `slurm_stacked_selector_baselines.sh`, `slurm_matched_baseline_figure.sh` and the comparison launchers of section 3.3 | `masked_f1_units.py`, `compute_matched_baseline_f1_curves.py`, `combine_mlp_baseline_curves.py`, `plot_selector_f1_fillrate.py` | `figures/f1_fillrate_3panel_oup.png`, `selector_f1_fillrate_mlp_baselines_1000_points.csv` | 3.3, 4 |
| Table 1 | Paired masked F1 differences and win counts | as Figure 1 | `stacked_selector_baselines.py`, `paired_masked_f1_bootstrap.py`, `combine_mlp_baseline_curves.py` | `masked_f1_paired_bootstrap.{csv,json}`, `selector_f1_fillrate_mlp_baselines_summary.json`, `stacked_selector_baselines/` | 4 |
| Table 2 | Accuracy of the inserted value | `slurm_safe_fusion_stack.sh` (`VALUE_MODEL=linear`), `slurm_autoencoder_fusion.sh`, `slurm_value_accuracy.sh` | `run_leakage_safe_method.py`, `run_autoencoder_fusion.py`, `fusion/`, `evaluate_value_accuracy.py` | `value_accuracy/error_removed.csv`, `value_accuracy/log_error.csv`, `value_accuracy/cross_validation.csv`, `value_accuracy/summary.json` | 8 |
| Table 3 | Zeros created by knockdown, with and without perturbation labels | `slurm_condition_aware_screens.sh`, the deployment launchers, `slurm_evaluate_perturbation_zeros.sh` | `evaluate_perturbation_zeros.py`, `evaluate_condition_masked_f1.py`, `apply_fill_fraction.py` | `perturbation_zeros/report.json`, `perturbation_zeros/per_target.csv`, `perturbation_zeros/masked_f1_report.json` | 9, 11 |
| Table 4 | Filling recorded zeros in held-out cells | `slurm_deployment_prepare.sh`, `slurm_deployment_scvi.sh`, `slurm_deployment_fill.sh`, `slurm_deployment_evaluate.sh` | `build_deployment_inputs.py`, `finalize_deployment_contracts.py`, the downstream evaluators, `summarize_fill_evaluations.py` | `downstream_deployment/summary_key_metrics.csv`, `downstream_deployment/evaluation/` | 9 |
| Figure 2 | CD274 RNA zeros against surface PD-L1 | `slurm_prepare_papalexi_crossmodal.sh`, `slurm_fetch_papalexi_multimodal.sh`, `slurm_audit_papalexi_multimodal.sh`, `slurm_papalexi_crossmodal_mlp.sh`, `slurm_papalexi_crossmodal_scvi.sh`, `slurm_evaluate_papalexi_cd274.sh`, `slurm_compare_method_rankings.sh` | `build_papalexi_crossmodal_benchmark.py`, `audit_papalexi_multimodal.py`, `evaluate_papalexi_crossmodal.py`, `evaluate_pdl1_state_baselines.py`, `plot_biological_range_figures.py`, `compare_method_rankings.py` | `figures/pdl1_range_validation.{png,pdf}`, `papalexi_crossmodal/benchmark/evaluation_cd274/`, `papalexi_crossmodal/benchmark/evaluation_pdl1_state/`, `ranking_comparison/cd274_report.json` | 2.6, 12 |
| Table S1 | Fill decisions of standard imputers | `slurm_standard_imputers.sh`, `slurm_evaluate_fill_decisions.sh` | `standard_imputers/run_standard_imputers.py`, `evaluate_fill_decisions.py` | `standard_imputers/`, `fill_decisions/table_s1.csv` | 14 |
| Table S2 | Benchmark data | `slurm_summarize_benchmark_data.sh` | `summarize_benchmark_data.py` | `benchmark_data/benchmark_data.csv` | 15 |
| Table S3 | Comparison methods | The comparison launchers of section 3.3 | `run_alra_baseline.py`, `run_saver_baseline.py`, `run_saver_baseline.R`, `run_magic_baseline.py`, `run_scvi_baseline.py`, `run_scgpt_gene_prediction.py`, `run_scgcl_baseline.py`, `evaluate_scgcl_baseline.py` | `baselines/`, `pancreas_crossfit/fold_<k>/alra` | 3.3, 16 |
| Table S4 | Mask and seed replicates | `slurm_seed_replicates.sh` | `make_mask_replicate.py`, `summarize_seed_replicates.py` | `seed_replicates/summary/` | 5 |
| Table S5 | Selector feature groups | `slurm_selector_mlp_attribution_range.sh`, `slurm_pair_mlp_attribution_range.sh` | `selector_attribution.py`, `combine_selector_attribution.py` | `selector_mlp_attribution_range/*/ranking_metrics.parquet` | 6 |
| Table S6 | Binomial thinning | `slurm_thinning_benchmark.sh` | `prepare_thinning_benchmark.py`, `evaluate_thinning_benchmark.py` | `thinning/evaluation/matched_fraction_summary.{csv,json}` | 7 |
| Table S7 | Inserted value by fill fraction and stratum | as Table 2, with `slurm_seed_replicates.sh` and `slurm_thinning_benchmark.sh` (`stack` with `VALUE_MODEL=linear`) | `evaluate_value_accuracy.py` | `value_accuracy/error_removed.csv`, `value_accuracy/log_error_strata.csv`, `value_accuracy/replicates.csv` | 8 |
| Table S8 | Knockdown AUROC by screen | as Table 3 | `evaluate_perturbation_zeros.py` | `perturbation_zeros/report.json` (`auroc_by_screen`) | 11 |
| Table S9 | RNA-protein agreement by replicate, raw counts and other proteins | as Figure 2, with `slurm_evaluate_papalexi_crossmodal.sh` and `slurm_evaluate_papalexi_crossmodal_raw.sh` | `evaluate_papalexi_crossmodal.py` | `papalexi_crossmodal/benchmark/{evaluation_cd274,evaluation_cd274_raw_counts,evaluation,evaluation_raw_counts}/` | 12 |
| Table S10 | Downstream endpoints on the masked benchmark | `slurm_apply_fill_fraction.sh`, `slurm_evaluate_complete_downstream.sh`, `slurm_finalize_complete_downstream.sh`, `slurm_decompose_fills.sh` | `evaluate_colon_donor_biology.py`, `evaluate_pancreas_crossfit_biology.py`, `evaluate_unsupervised_clustering.py`, `combine_clustering_folds.py`, `evaluate_trajectory_preservation.py`, `evaluate_interventional_grn.py`, `summarize_complete_downstream.py`, `plot_complete_downstream.py`, `verify_complete_downstream.py`, `decompose_fills.py`, `summarize_fill_evaluations.py` | `downstream_complete/summary/`, `downstream_decomposition/summary.csv` | 13 |
| Table S11 | Fill-fraction rule | `slurm_detection_rule.sh` | `calibrated_selective_fill.py` (`--detection-rule-mask-rate`) | `detection_rule/<unit>/calibration_report.json`, `downstream_deployment/<unit>/detection_rule/calibration_report.json` | 10 |
| Table S12 | Runtime | none | `summarize_runtime.py`, `runtime_jobs.tsv` | `runtime/runtime_table.csv` | 17 |
