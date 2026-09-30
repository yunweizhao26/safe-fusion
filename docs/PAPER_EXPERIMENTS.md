# Manuscript items, scripts and outputs

The main method is transductive Safe Fusion. The inductive mode is reported
separately. [REPRODUCTION.md](REPRODUCTION.md) gives prerequisites, exact commands
and output files for every row below. Its sections 1–18 build shared reference
inputs; sections 19–26 build the final analyses.

Output paths below are relative to `artifacts/paper_evidence/`. `R2`, `R3` and
`R4` mean `review_round2`, `review_round3` and `review_round4`. Dataset and tissue
placeholders expand as listed in the reproduction guide.

| Manuscript item | Content | Section | Output files |
|---|---|---|---|
| Table 1 | Masked recovery | 21 | `R4/transductive_main/masked/<dataset>/{paired_differences,new_comparators_paired_differences}.csv` |
| Table 2 | Zeros with known status | 22, 23 | `R4/transductive_references/knockdown/evaluation_matched/report.json; R3/comparators/evaluation/known_zeros/knockdown/auroc.csv; R4/sex_zero_comparators/reproduction_run/evaluation/<tissue>/auroc.csv` |
| Table 3 | Error against deeper counts | 23 | `R4/transductive_downstream/summary/thinning_reference.csv` |
| Figure 1 | Overview and masked F1 | 21 | `R4/transductive_main/figures/f1_fillrate_3panel_oup.png` |
| Figure 2 | CD274 RNA and PD-L1 protein | 22 | `R4/transductive_references/protein_cs/figures/pdl1_range_validation.png` |
| Supplementary Table S1 | Standard-imputer fill rates | 15 | `fill_decisions/table_s1.csv` |
| Supplementary Table S2 | Benchmark data | 19 | `benchmark_data/benchmark_data.csv; R2/leakage_free/benchmark_data/benchmark_data.csv; R3/colon_crossfit/benchmark_data.csv` |
| Supplementary Table S3 | Comparison settings | 17, 19 | `Comparison launcher settings and each ALRA metadata.json chosen_k` |
| Supplementary Table S4 | Mask and model replicates | 21 | `R4/transductive_main/rebuilt_replicates/evaluation/{across_seeds,paired_differences,wins}.csv` |
| Supplementary Table S5 | All comparison rankings | 21 | `R4/transductive_main/masked/<dataset>/{paired_differences,new_comparators_paired_differences}.csv` |
| Supplementary Table S6 | Teacher and classifier ablations | 21 | `R4/transductive_main/item3_evaluation/<dataset>/paired_differences.csv` |
| Supplementary Table S7 | Selector feature groups | 21 | `R4/transductive_main/item3_evaluation/feature_pr_auc_exact.csv` |
| Supplementary Table S8 | Inductive settings | 20 | `R3/value_v2_ablations/ablations/evaluation/paired_differences.csv` |
| Supplementary Table S9 | Alternative training targets | 20, 25 | `R3/selector_v2/test/evaluation/{masked,thinned}/paired_differences.csv and references/{knockdown_summary,sex_paired,pdl1_paired}.csv; R4/dropout_posterior/evaluation/{masked,thinning,known,sex,protein}/ endpoint results.csv files` |
| Supplementary Table S10 | Binomial thinning | 21 | `R4/transductive_main/thinning_evaluation/{absolute,paired_differences}.csv` |
| Supplementary Table S11 | Recall by count | 8.3 | `R2/thinning_transfer/count_stratified_recall/count_stratified_recall.csv` |
| Supplementary Table S12 | Inserted-value accuracy | 9 | `R2/norman_rebuilt/value_accuracy/{error_removed,log_error,cross_validation}.csv` |
| Supplementary Table S13 | Value accuracy by budget and stratum | 9 | `R2/norman_rebuilt/value_accuracy/{error_removed,log_error_strata,replicates}.csv` |
| Supplementary Table S14 | Values at recorded zeros | 11 | `R2/inserted_value/recorded_zero_fills.csv` |
| Supplementary Table S15 | Thinning value bias | 11, 20 | `R2/inserted_value/thinning_positive_bias.csv; R3/value_v2_ablations/value/test/value_accuracy.csv` |
| Supplementary Table S16 | Perturbation zero controls | 22 | `R4/transductive_references/knockdown/evaluation_matched/{report.json,effects.csv,auroc.csv}` |
| Supplementary Table S17 | Masked F1 per screen | 22 | `R4/transductive_references/knockdown/masked_f1/masked_f1_by_screen.json` |
| Supplementary Table S18 | Sex-linked zero controls | 22, 23 | `R4/transductive_references/sex_zeros/evaluation/<tissue>/auroc.csv; R4/sex_zero_comparators/reproduction_run/evaluation/<tissue>/auroc.csv` |
| Supplementary Table S19 | Protein agreement | 22 | `R4/transductive_references/protein_cs/{evaluation_cd274/replicate_association,evaluation_cd274_raw_counts/continuous_protein_association,evaluation/continuous_protein_association}.csv` |
| Supplementary Table S20 | Protein agreement within state | 22 | `R4/transductive_references/protein_cs/evaluation_within_state/{association,pdl1_pooled_rankings}.csv` |
| Supplementary Table S21 | Masked downstream endpoints | 23 | `R4/transductive_downstream/summary/{s17_absolute,endpoint_changes,decomposition_counts}.csv` |
| Supplementary Table S22 | Recorded-count downstream endpoints | 23 | `R4/transductive_downstream/summary/endpoint_changes.csv` |
| Supplementary Table S23 | Disease effects and null tests | 23 | `R4/transductive_downstream/summary/disease_all.csv` |
| Supplementary Table S24 | Reference mapping changes | 23 | `R4/transductive_downstream/disease/<tissue>_<fold>/deployment/annotation/<tissue>/overall.csv` |
| Supplementary Table S25 | Correlation null | 23 | `R4/transductive_downstream/correlation/null/counts/{summary,versus_reference}.csv` |
| Supplementary Table S26 | Fill-fraction rule | 10.3, 10.4 | `detection_rule/<unit>/calibration_report.json; downstream_deployment/<unit>/detection_rule/calibration_report.json; R2/norman_rebuilt/norman_downstream_rows.csv` |
| Supplementary Table S27 | Runtime | 18 | `runtime/runtime_table.csv` |
| Supplementary Table S28 | Scaling study | 26 | `R3/scale/results/{paired_differences,resources,safe_fusion_totals}.csv` |
| Supplementary Table S29 | Lupus Treg example | 24 | `R4/sle_transductive/<value>/results/{masked/paired_bootstrap,q1_zeros_sex/per_gene,q3_clustering/treg_calls_summary,q3_clustering/clustering_mean_over_folds,focus_fills/focus_fill_counts,deploy_main/q4_effects,deploy_main/q4_null_de,q7_protein/fcrl3_protein_agreement}.csv; corresponding sle_treg_case/results/ files` |

| Section | Source entry points | Output root under `artifacts/paper_evidence/` |
|---|---|---|
| 15 | `scripts/slurm_standard_imputers.sh`, `scripts/evaluate_fill_decisions.py` | `fill_decisions/table_s1.csv` |
| 19 | `scripts/slurm_colon_crossfit.sh`, `scripts/slurm_leakage_free_rerun.sh`, `scripts/slurm_r3_comparators.sh`, `scripts/slurm_r3_comparators_evaluate.sh`, `scripts/summarize_benchmark_data.py` | `review_round3/colon_crossfit/`, `review_round2/leakage_free/`, `review_round3/comparators/` |
| 20 | `scripts/slurm_v2_ablation.sh`, `scripts/slurm_v2_selector.sh`, `scripts/slurm_v2_selector_test.sh`, `scripts/slurm_v2_value.sh` | `review_round3/value_v2_ablations/`, `review_round3/selector_v2/` |
| 21 | `scripts/analyses/transductive_main/run.sh`, `scripts/slurm_thinning_transductive.sh`, `scripts/build_transductive_fillrate_summary.py`, `scripts/plot_overview_f1.py` | `review_round4/transductive_main/` |
| 22 | `scripts/analyses/transductive_references/reproduce.sh` | `review_round4/transductive_references/` |
| 23 | `scripts/analyses/transductive_downstream/reproduce_all.sh`, `scripts/analyses/sex_zero_comparators/reproduce.sh` | `review_round4/transductive_downstream/`, `review_round4/sex_zero_comparators/reproduction_run/` |
| 24 | `scripts/slurm_sle_pipeline.sh`, `scripts/analyses/sle_transductive/reproduce.sh` | `sle_treg_case/`, `review_round4/sle_transductive/` |
| 25 | `scripts/analyses/dropout_posterior/run.sh`, `scripts/analyses/dropout_posterior/evaluate.sh` | `review_round4/dropout_posterior/evaluation/` |
| 26 | `scripts/slurm_scale.sh` | `review_round3/scale/` |
| 10.3–10.4 | `scripts/slurm_detection_rule.sh`, `scripts/slurm_norman_rebuilt_downstream.sh` | `detection_rule/`, `downstream_deployment/`, `review_round2/norman_rebuilt/` |
| 18 | `scripts/summarize_runtime.py` | `runtime/runtime_table.csv` |

Table S3 records method settings rather than computed estimates. Runtime and
scale costs depend on Slurm accounting from the measured run. Unavailable
EnImpute sex endpoints remain unavailable in the manuscript; incomplete folds
must not be substituted for complete cohorts. Table S9's omitted inductive
Norman thinning posterior is not needed for its reported endpoints.
