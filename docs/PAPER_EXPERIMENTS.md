# Manuscript items, scripts and outputs

The main method is transductive Safe Fusion. The inductive mode is reported
separately. [REPRODUCTION.md](REPRODUCTION.md) gives exact commands and their
dependency order. Sections 1–18 build shared reference inputs, sections 19–26
build the earlier final analyses, and sections 27–35 add the final revision.
The 29 supplementary table labels and Supplementary Figure S1 follow their
order in the final `supplementary_results.tex`.

Output paths below are relative to `artifacts/paper_evidence/`. `R2`, `R3` and
`R4` mean `review_round2`, `review_round3` and `review_round4`. Paths following a
semicolon retain the analysis root of the preceding path unless written in full.

| Manuscript item | Content | Section | Output files |
|---|---|---|---|
| Table 1 | Masked recovery | 21, 27, 29 | `R4/transductive_comparators/paired_differences.csv`; `R4/transductive_main/masked/<dataset>/{paired_differences,new_comparators_paired_differences}.csv`; `R4/round2_extras/part_b/highest_f1.csv` |
| Table 2 | Zeros with known status | 22, 23, 28, 29 | `R4/round2_extras/part_a/table2_aurocs.csv` (continuous estimates and intervals, including label-aware rows) |
| Table 3 | Error against deeper counts | 23, 31 | `R4/reviewer_extras/B_thinning/{all_rows,table3_new_rows_x100}.csv` |
| Figure 1 | Overview and masked F1 | 21, 27, 29 | `R4/round2_extras/part_b/selector_f1_fillrate_allcell_1000_points.csv`; `R4/transductive_main/figures/f1_fillrate_3panel_oup.png` |
| Figure 2 | CD274 RNA and PD-L1 protein | 22 | `R4/transductive_references/protein_cs/figures/pdl1_range_validation.png` |
| 22.1 | `scripts/analyses/protein_full/submit.sh`, `scripts/analyses/protein_full/stage.sh`, `scripts/analyses/protein_full/prepare.py`, `scripts/analyses/protein_full/selector.py`, `scripts/analyses/protein_full/evaluate_full.py`, `scripts/analyses/protein_full/evaluate.py` | `review_round4/protein_full/full/`, `review_round4/protein_followup/full/` |
| Supplementary Figure S1 | Downstream error across fill fractions | 23, 31, 32 | `R4/fill_grid/fill_grid_long.csv`; `R4/fill_grid/fill_grid.{pdf,png}` |
| Supplementary Table S1 | Standard-imputer fill rates | 15 | `fill_decisions/table_s1.csv` |
| Supplementary Table S2 | Benchmark data | 19 | `benchmark_data/benchmark_data.csv; R2/leakage_free/benchmark_data/benchmark_data.csv; R3/colon_crossfit/benchmark_data.csv` |
| Supplementary Table S3 | Comparison settings | 17, 19 | `Comparison launcher settings and each ALRA metadata.json chosen_k` |
| Supplementary Table S4 | Mask and model replicates | 21, 29 | `R4/transductive_main/rebuilt_replicates/evaluation/{across_seeds,paired_differences,wins}.csv`; `R4/round2_extras/part_c/{across_seeds,paired_per_seed}.csv` |
| Supplementary Table S5 | All comparison rankings | 21, 27 | `R4/transductive_main/masked/<dataset>/{paired_differences,new_comparators_paired_differences}.csv`; `R4/transductive_comparators/{absolute,paired_differences}.csv` |
| Supplementary Table S6 | Teacher and classifier ablations | 21, 28 | `R4/transductive_main/item3_evaluation/<dataset>/paired_differences.csv`; label-baseline text: `R4/label_baselines/masked/<dataset>/report.json` and `masked/supplementary/<dataset>/report.json` |
| Supplementary Table S7 | Selector feature groups | 21 | `R4/transductive_main/item3_evaluation/feature_pr_auc_exact.csv` |
| Supplementary Table S8 | Inductive settings | 20 | `R3/value_v2_ablations/ablations/evaluation/paired_differences.csv` |
| Supplementary Table S9 | Alternative training targets | 20, 25 | `R3/selector_v2/test/evaluation/{masked,thinned}/paired_differences.csv and references/{knockdown_summary,sex_paired,pdl1_paired}.csv; R4/dropout_posterior/evaluation/{masked,thinning,known,sex,protein}/ endpoint results.csv files` |
| Supplementary Table S10 | Binomial thinning | 21 | `R4/transductive_main/thinning_evaluation/{absolute,paired_differences}.csv` |
| Supplementary Table S11 | Recall by count | 8.3 | `R2/thinning_transfer/count_stratified_recall/count_stratified_recall.csv` |
| Supplementary Table S12 | Inserted-value accuracy | 9, 30 | `R4/current_tables/current/masked/{error_removed,log_error}.csv`; historical autoencoders: `R2/norman_rebuilt/value_accuracy/{error_removed,log_error}.csv` |
| Supplementary Table S13 | Value accuracy by budget and stratum | 9, 30 | `R4/current_tables/current/masked/{error_removed,log_error_strata}.csv`; historical autoencoders: `R2/norman_rebuilt/value_accuracy/{error_removed,log_error_strata}.csv` |
| Supplementary Table S14 | Values at recorded zeros | 30 | `R4/current_tables/current/recorded/recorded_zero_fills.csv` |
| Supplementary Table S15 | Thinning value bias | 30 | `R4/current_tables/current/thinning/bias.csv` |
| Supplementary Table S16 | Perturbation zero controls | 22, 28 | `R4/label_baselines/screens/verification_transductive/{auroc,effects}.csv`; `screens/verification_inductive_comparators/knockdown/{auroc,effects}.csv`; `screens/s16_paper_cells.csv` |
| Supplementary Table S17 | Masked F1 per screen | 22 | `R4/transductive_references/knockdown/masked_f1/masked_f1_by_screen.json` |
| Supplementary Table S18 | Sex-linked zero controls | 22, 23, 28 | `R4/sex_zero_comparators/reproduction_run/evaluation/<tissue>/auroc.csv`; sensitivity text: `R4/label_baselines/sex/colon/{agreed25,metadata34}_auroc.csv` |
| Supplementary Table S19 | Protein agreement | 22 | `R4/transductive_references/protein_cs/{evaluation_cd274/replicate_association,evaluation_cd274_raw_counts/continuous_protein_association,evaluation/continuous_protein_association}.csv` |
| Supplementary Table S20 | Protein agreement within state | 22 | `R4/transductive_references/protein_cs/evaluation_within_state/{association,pdl1_pooled_rankings}.csv` |
| Supplementary Section S7, Full screen | PD-L1 across all 20,156 Papalexi cells | 22.1 | `R4/protein_followup/full/evaluation/{association,paired_differences,pdl1_pooled_rankings,global_fill}.csv` |
| Supplementary Table S21 | Masked downstream endpoints | 23 | `R4/transductive_downstream/summary/{s17_absolute,endpoint_changes,decomposition_counts}.csv` |
| Supplementary Table S22 | Recorded-count downstream endpoints | 23 | `R4/transductive_downstream/summary/endpoint_changes.csv` |
| Supplementary Table S23 | Disease effects and null tests | 23 | `R4/transductive_downstream/summary/disease_all.csv` |
| Supplementary Table S24 | Reference mapping changes | 23 | `R4/transductive_downstream/disease/<tissue>_<fold>/deployment/annotation/<tissue>/overall.csv` |
| Supplementary Table S25 | Correlation null | 23 | `R4/transductive_downstream/correlation/null/counts/{summary,versus_reference}.csv` |
| Supplementary Table S26 | Fill-fraction rule | 30 | `R4/current_tables/current/rule/summary.csv`; `current/rule/{masked,recorded}/<unit>/calibration_report.json` |
| Supplementary Table S27 | Runtime | 18 | `runtime/runtime_table.csv` |
| Supplementary Table S28 | Scaling study | 26, 31, 33, 34 | `R3/scale/results/{paired_differences,resources,safe_fusion_totals}.csv`; Section S8 extensions: `R4/reviewer_extras/C_scale/results/paired_differences.csv`, `R4/scale_caps/genes_<2000 or 5000>/evaluation/results/paired_differences.csv`, `R4/scale_comparators/{paired_differences,resources}.csv` |
| Supplementary Table S29 | Lupus Treg example | 24, 35 | `R4/sle_donor_labels/comparisons/masked_f1_intervals.csv`; `comparisons/{q1_zeros_sex/per_gene,q3_clustering/treg_calls_summary,q3_clustering/clustering_mean_over_folds,focus_fills/focus_fill_counts,deploy_main/q5_modules,deploy_main/q4_effects,deploy_main/q4_null_de,q7_protein/fcrl3_protein_agreement}.csv` |

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
| 27 | `scripts/analyses/transductive_comparators/submit.sh` | `review_round4/transductive_comparators/` |
| 28 | `scripts/analyses/label_baselines/reproduce.sh` | `review_round4/label_baselines/` |
| 29 | `scripts/analyses/round2_extras/reproduce.sh` | `review_round4/round2_extras/` |
| 30 | `scripts/analyses/current_tables/run.sh` | `review_round4/current_tables/` |
| 31 | `scripts/analyses/reviewer_extras/B_thinning/submit.sh; scripts/analyses/reviewer_extras/C_scale/code/submit.sh` | `review_round4/reviewer_extras/` |
| 32 | `scripts/analyses/fill_grid/submit.sh` | `review_round4/fill_grid/` |
| 33 | `scripts/analyses/scale_caps/submit.sh` | `review_round4/scale_caps/` |
| 34 | `scripts/analyses/scale_comparators/submit.py; scripts/analyses/scale_comparators/monitor.sh` | `review_round4/scale_comparators/` |
| 35 | `scripts/analyses/sle_donor_labels/reproduce.sh` | `review_round4/sle_donor_labels/` |

Section S8's calibration text reads `R4/reviewer_extras/A_calibration/calibration.csv`
(section 31). Its 5,000-gene, training-cap and additional-comparator text uses
sections 31, 33 and 34. Section S9 and S29 include both donor-label values.
Figure 2 retains the 8.0 by 1.9 inch layout and the label
“CD274 RNA zeros selected (%)”.

Table S3 records settings rather than numerical estimates. Historical runtime
values require the original accounting; new runs measure their own costs.
The historical print audits require external manuscript inputs listed in
section 27. They are not bundled or silently replaced by the current paper.
The unavailable transductive autoencoder rows and Dixit S14 inputs remain
explicit. Table S9's omitted inductive Norman thinning posterior is not needed
for its reported endpoints.
