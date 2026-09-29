# Manuscript items, scripts and outputs

This table maps every table and figure of the manuscript and its supplement
to the launchers and scripts that produce it and to its output files. Every
launcher is in `scripts/`. Output paths are under `artifacts/paper_evidence/`
unless they start with `artifacts/` or `<methods_root>`, and `rr2/` stands for
`artifacts/paper_evidence/review_round2/`. `<methods_root>` is the contract
directory of a dataset in `scripts/unit_paths.sh`.
[REPRODUCTION.md](REPRODUCTION.md) gives the commands and their dependencies,
and the section column refers to it. Every item also needs the data of
section 2 and the fits of section 3: the production teachers, fused value and
selectors (`slurm_prepare_complete_downstream_teachers.sh`,
`slurm_scvi_teachers.sh`, `slurm_safe_fusion_stack.sh`,
`slurm_complete_downstream_selectors.sh`) and the leakage-free build of the
pancreas folds and the Norman screen (`slurm_leakage_free_rerun.sh`).

| Item | Content | Launchers | Scripts | Outputs | Section |
|---|---|---|---|---|---|
| Table 1 | Masked F1 of Safe Fusion minus each comparison, fractions with the highest F1 | `slurm_leakage_free_rerun.sh` (stages `alra`, `saver`, `magic`, `scvi`, `scgpt`, `stacked`, `evaluate`), `slurm_stacked_selector_baselines.sh`, `slurm_matched_baseline_curves.sh`, `slurm_fusion_value.sh` and the launchers of section 3.3 | `masked_f1_units.py`, `write_leakage_free_units.py`, `compute_matched_baseline_f1_curves.py`, `combine_mlp_baseline_curves.py`, `stacked_selector_baselines.py`, `paired_masked_f1_bootstrap.py`, `fusion_value_selectors.py`, `nonzero_probability.py`, `count_scale_contract.py`, `fusion_value_bootstrap.py` | `rr2/leakage_free/masked_f1_paired_bootstrap.{csv,json}`, `rr2/leakage_free/selector_f1_fillrate_mlp_baselines_summary.json`, `rr2/fusion_value/evaluation/summary_table.csv` | 3.4, 4, 5 |
| Table 2 | Zeros with known status: AUROC unadjusted and within library-size strata, masked F1 with and without labels | `slurm_condition_aware_screens.sh`, `slurm_knockdown_norman_rebuilt.sh`, `slurm_knockdown_standard_imputers.sh`, `slurm_knockdown_selector_scores.sh`, `slurm_papalexi_mixscape.sh`, `slurm_evaluate_knockdown_zero_analyses.sh`, the deployment launchers, `slurm_disease_control_analysis.sh` (Sex column) | `evaluate_knockdown_zero_analyses.py`, `evaluate_perturbation_zeros.py`, `evaluate_condition_masked_f1.py`, `run_papalexi_mixscape.py`, `disease_control_sex_zeros.py` | `rr2/knockdown/evaluation/report.json`, `rr2/knockdown/masked_f1/masked_f1_report.json`, `disease_control_checks/sex_zeros/pancreas/auroc.csv` | 10.1, 12, 13 |
| Table 3 | Filling recorded zeros in held-out cells | `slurm_deployment_prepare.sh`, `slurm_deployment_scvi.sh`, `slurm_deployment_fill.sh`, `slurm_deployment_evaluate.sh`, `slurm_norman_rebuilt_downstream.sh` (stages `deploy_*`, `summarize`) | `build_deployment_inputs.py`, `finalize_deployment_contracts.py`, the downstream evaluators, `summarize_fill_evaluations.py`, `summarize_norman_rebuilt_downstream.py` | `downstream_deployment/summary_key_metrics.csv`, `rr2/norman_rebuilt/norman_downstream_rows.csv` (rows with `table` 3) | 10.1, 10.4 |
| Figure 1 | Masked F1 at 1000 fill fractions | as Table 1 | `plot_selector_f1_fillrate.py` | `rr2/leakage_free/figures/f1_fillrate_3panel_oup.png`, `rr2/leakage_free/selector_f1_fillrate_mlp_baselines_1000_points.csv` | 4 |
| Figure 2 | CD274 RNA zeros against surface PD-L1 | `slurm_prepare_papalexi_crossmodal.sh`, `slurm_fetch_papalexi_multimodal.sh`, `slurm_audit_papalexi_multimodal.sh`, `slurm_papalexi_crossmodal_mlp.sh`, `slurm_papalexi_crossmodal_scvi.sh`, `slurm_evaluate_papalexi_cd274.sh` | `build_papalexi_crossmodal_benchmark.py`, `audit_papalexi_multimodal.py`, `evaluate_papalexi_crossmodal.py`, `evaluate_pdl1_state_baselines.py`, `plot_biological_range_figures.py` | `figures/pdl1_range_validation.{png,pdf}`, `papalexi_crossmodal/benchmark/evaluation_cd274/` | 2.7, 14 |
| Table S1 | Fill decisions of standard imputers | `slurm_standard_imputers.sh`, `slurm_evaluate_fill_decisions.sh` | `standard_imputers/run_standard_imputers.py`, `evaluate_fill_decisions.py` | `fill_decisions/table_s1.csv`, `fill_decisions/{gene_patterns,pair_agreement}.csv` | 15 |
| Table S2 | Benchmark data | `slurm_summarize_benchmark_data.sh`, `slurm_leakage_free_rerun.sh` (stage `benchmark_data`) | `summarize_benchmark_data.py` | `benchmark_data/benchmark_data.csv`, `rr2/leakage_free/benchmark_data/benchmark_data.csv` | 16 |
| Table S3, Section S3 | Comparison methods, scGCL | The comparison launchers of sections 3.3 and 3.4, `slurm_norman_rebuilt_supplement.sh` (stages `scgcl`, `scgcl_compare`) | `run_alra_baseline.py`, `run_saver_baseline.py`, `run_saver_baseline.R`, `run_magic_baseline.py`, `run_scvi_baseline.py`, `run_scgpt_gene_prediction.py`, `run_scgcl_baseline.py`, `evaluate_scgcl_baseline.py` | `baselines/`, `rr2/leakage_free/baselines/`, `masked_f1_paired_bootstrap.csv`, `rr2/norman_rebuilt/scgcl_comparison/` | 3.3, 3.4, 17 |
| Table S4 | Mask replicates and selectors trained on one method | `slurm_seed_replicates.sh`, `slurm_norman_rebuilt_replicates.sh`, as Table 1 for the lower block | `make_mask_replicate.py`, `summarize_seed_replicates.py`, `stacked_selector_baselines.py` | `seed_replicates/summary/`, `rr2/norman_rebuilt/seed_replicates/summary/`, `rr2/leakage_free/masked_f1_paired_bootstrap.csv` | 4, 6 |
| Table S5 | Sources of the ranking gain | `slurm_fusion_value.sh` | `run_leakage_safe_method.py --transductive`, `count_scale_contract.py`, `fusion_value_selectors.py`, `nonzero_probability.py`, `fusion_value_bootstrap.py` | `rr2/fusion_value/evaluation/{summary_table,paired_differences,absolute}.csv`, `rr2/fusion_value/evaluation/summary.json` | 5 |
| Table S6 | Selector feature groups | `slurm_selector_mlp_attribution_range.sh`, `slurm_pair_mlp_attribution_range.sh`, `slurm_norman_rebuilt_supplement.sh` (stage `attribution`) | `selector_attribution.py`, `combine_selector_attribution.py` | `selector_mlp_attribution_range/*/ranking_metrics.parquet`, `rr2/norman_rebuilt/selector_mlp_attribution_range/norman_crispra/ranking_metrics.parquet` | 7 |
| Table S7 | Binomial thinning | `slurm_thinning_benchmark.sh`, `slurm_thinning_transfer.sh` | `prepare_thinning_benchmark.py`, `evaluate_thinning_benchmark.py`, `prepare_thinning_transfer.py`, `stacked_selector_scores.py`, `evaluate_thinning_transfer.py` | `rr2/thinning_transfer/evaluation/transfer_summary.{csv,json}`, `rr2/thinning_transfer/evaluation/transfer_paired_differences.csv`, `thinning/evaluation/matched_fraction_summary.csv` | 8.1, 8.2 |
| Table S8 | Recall by count stratum | `slurm_count_stratified_recall.sh` | `stacked_selector_scores.py`, `evaluate_count_stratified_recall.py` | `rr2/thinning_transfer/count_stratified_recall/count_stratified_recall.{csv,json}` | 8.3 |
| Table S9 | Accuracy of the inserted value | `slurm_safe_fusion_stack.sh` (`VALUE_MODEL=linear`), `slurm_autoencoder_fusion.sh`, `slurm_norman_rebuilt_value.sh` | `run_leakage_safe_method.py`, `run_autoencoder_fusion.py`, `fusion/`, `evaluate_value_accuracy.py` | `rr2/norman_rebuilt/value_accuracy/{error_removed,log_error,cross_validation}.csv`, `rr2/norman_rebuilt/value_accuracy/summary.json` | 9 |
| Table S10 | Inserted value by fill fraction and stratum | as Table S9, with `slurm_seed_replicates.sh`, `slurm_norman_rebuilt_replicates.sh` and `slurm_thinning_benchmark.sh` (`stack` with `VALUE_MODEL=linear`) | `evaluate_value_accuracy.py` | `rr2/norman_rebuilt/value_accuracy/{error_removed,log_error_strata,replicates}.csv` | 9 |
| Table S11 | Values inserted into recorded zeros | `slurm_inserted_value.sh` and the deployment launchers | `evaluate_inserted_value.py` | `rr2/inserted_value/recorded_zero_fills.csv` | 11 |
| Table S12 | Bias of inserted values at thinned entries | `slurm_inserted_value.sh`, `slurm_thinning_transfer.sh`, `slurm_thinning_benchmark.sh` | `evaluate_inserted_value.py` | `rr2/inserted_value/thinning_positive_bias.csv` | 11 |
| Table S13 | Zeros with known status by screen and fold changes after filling | as Table 2 | `evaluate_knockdown_zero_analyses.py` | `rr2/knockdown/evaluation/report.json` and its CSV files | 13 |
| Table S14 | Zeros of sex-linked genes | `slurm_disease_control_fill.sh`, `slurm_disease_control_analysis.sh` | `disease_control_common.py`, `disease_control_sex_zeros.py` | `disease_control_checks/sex_zeros/<tissue>/auroc.csv` | 12 |
| Table S15 | RNA-protein agreement by replicate, raw counts and other proteins | as Figure 2, with `slurm_evaluate_papalexi_crossmodal.sh` and `slurm_evaluate_papalexi_crossmodal_raw.sh` | `evaluate_papalexi_crossmodal.py` | `papalexi_crossmodal/benchmark/{evaluation_cd274,evaluation_cd274_raw_counts,evaluation,evaluation_raw_counts}/` | 14 |
| Table S16 | RNA-protein agreement beyond perturbation state | `slurm_evaluate_protein_within_state.sh` | `calibrated_selective_fill.py`, `evaluate_protein_within_state.py` | `rr2/protein/evaluation/{pdl1_pooled_rankings,association}.csv` | 14 |
| Table S17 | Downstream endpoints on the masked benchmark | `slurm_apply_fill_fraction.sh`, `slurm_evaluate_complete_downstream.sh`, `slurm_finalize_complete_downstream.sh`, `slurm_decompose_fills.sh`, `slurm_norman_rebuilt_downstream.sh` (stages `matched`, `masked_grn`, `summarize`) | `evaluate_colon_donor_biology.py`, `evaluate_pancreas_crossfit_biology.py`, `evaluate_unsupervised_clustering.py`, `combine_clustering_folds.py`, `evaluate_trajectory_preservation.py`, `evaluate_interventional_grn.py`, `summarize_complete_downstream.py`, `plot_complete_downstream.py`, `verify_complete_downstream.py`, `decompose_fills.py`, `summarize_fill_evaluations.py` | `downstream_complete/summary/all_bootstrap_summaries.csv`, `rr2/norman_rebuilt/norman_downstream_rows.csv` (rows with `table` S17), `downstream_decomposition/summary.csv` | 10.2, 10.4 |
| Table S18 | Disease effects on recorded counts | `slurm_disease_control_fill.sh`, `slurm_disease_control_analysis.sh` | `disease_control_effects.py` | `disease_control_checks/disease_effects/<tissue>/{observed_summary,permutation_null}.csv` | 12 |
| Table S19 | Module scores | as Table S18 | `disease_control_modules.py` | `disease_control_checks/module_scores/<tissue>/disease_effect.csv` | 12 |
| Table S20 | Reference mapping labels | as Table S18 | `disease_control_annotation.py` | `disease_control_checks/annotation/<tissue>/reference_mapping_overall.csv` | 12 |
| Table S21 | Fill-fraction rule | `slurm_detection_rule.sh`, `slurm_norman_rebuilt_downstream.sh` (stages `detection_masked`, `detection_deploy`, `summarize`) | `calibrated_selective_fill.py` (`--detection-rule-mask-rate`) | `detection_rule/<unit>/calibration_report.json`, `downstream_deployment/<unit>/detection_rule/calibration_report.json`, `rr2/norman_rebuilt/norman_downstream_rows.csv` (rows with `table` S21) | 10.3, 10.4 |
| Table S22 | Runtime | none | `summarize_runtime.py`, `runtime_jobs.tsv` | `runtime/runtime_table.csv` | 18 |

Some statements of the text come from outputs that no table uses:

- The effect of the pancreas gene set on Table 1 (Methods) is the
  `change_pp` column of `rr2/leakage_free/table1_old_vs_new.csv`
  (`compare_leakage_free_benchmarks.py`, section 4).
- The Norman condition counts (Methods) are in
  `rr2/leakage_free/norman_crispra/prepared.report.json` (section 2.4).
- The Dixit fold change of Supplementary Section S6 is in
  `rr2/label_audit/label_audit.json` (`audit_prepared_labels.py`, section
  2.8).
- The random-label null test of Supplementary Section S6 is in
  `rr2/knockdown/pseudolabel_null/adamson_crispri/evaluation/`
  (`slurm_knockdown_pseudolabel_null.sh`, `knockdown_pseudolabel_null.py`,
  section 13.2).
- The shares of masked positives among the filled entries (Results) are in
  `downstream_decomposition/<dataset>/safe_fusion/decomposition_counts.csv`
  (section 10.2).
