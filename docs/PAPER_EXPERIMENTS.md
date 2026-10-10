# Manuscript items, scripts and outputs

The main method is transductive Safe Fusion. The inductive mode is reported
separately. [REPRODUCTION.md](REPRODUCTION.md) gives exact commands and their
dependency order. Sections 1–18 build shared reference inputs, sections 19–26
build the earlier final analyses, and sections 27–44 add the final revision.
Tables 1–3, Figures 1–3, Supplementary Figure S1 and Supplementary Tables S1–S34
follow their order in the final `supplementary_results.tex`.

Output paths below are relative to `artifacts/paper_evidence/`. `R2`, `R3` and
`R4` mean `review_round2`, `review_round3` and `review_round4`. Paths following a
semicolon retain the analysis root of the preceding path unless written in full.

| Manuscript item | Content | Section | Output files |
|---|---|---|---|
| Table 1 | Mean masked F1 ± bootstrap SE, all methods | 38 | `R4/main_tables_v3/estimates.csv`; labeled rows originate in `R4/main_tables_v2/estimates.csv` |
| Table 2 | Known-zero AUROC ± bootstrap SE, all nine published methods and teacher ablations | 38, 44 | `R4/main_tables_v3/estimates.csv`; `R4/table2_added/table2_rows.csv` |
| Table 3 | CD274 zero rankings against PD-L1, ten published rankings on both screen sizes | 37, 39, 42, 43 | `R4/published_protein/{estimates,table3_rows}.csv` |
| Figure 1 | Method schematic | 41 | `scripts/plot_method_schematic.py`; `artifacts/paper_evidence/figures/safe_fusion_method.{pdf,png,svg}` |
| Figure 2 | Masked F1 at matched fill fractions | 29, 41 | `R4/round2_extras/part_b/selector_f1_fillrate_allcell_1000_points.csv`; `artifacts/paper_evidence/figures/overview_f1_fillrate.{pdf,png}` |
| Supplementary Figure S1 | CD274 RNA and PD-L1 protein | 22 | `R4/transductive_references/protein_cs/figures/pdl1_range_validation.png` |
| Figure 3 | Gap to unthinned counts closed, 19 fill fractions, all count-valued published methods | 32, 40, 42 | `R4/published_downstream/fill_grid_published_long.csv`; `R4/published_downstream_allcells/fill_grid_allcells_long.csv`; `artifacts/paper_evidence/figures/fill_grid_main.{pdf,png}` |
| Supplementary Table S1 | Standard-imputer fill rates | 15 | `fill_decisions/table_s1.csv` |
| Supplementary Table S2 | Benchmark data | 19 | `benchmark_data/benchmark_data.csv; R2/leakage_free/benchmark_data/benchmark_data.csv; R3/colon_crossfit/benchmark_data.csv` |
| Supplementary Table S3 | Comparison settings | 17, 19 | `Comparison launcher settings and each ALRA metadata.json chosen_k` |
| Supplementary Table S4 | Full masked recovery: paired differences, probability, selectors and inductive mode | 21, 27, 29, 38 | `R4/transductive_main/masked/<dataset>/{paired_differences,new_comparators_paired_differences}.csv`; `R4/transductive_comparators/paired_differences.csv`; `R4/round2_extras/part_b/highest_f1.csv` |
| Supplementary Table S5 | Mask and model replicates | 21, 29 | `R4/transductive_main/rebuilt_replicates/evaluation/{across_seeds,paired_differences,wins}.csv`; `R4/round2_extras/part_c/{across_seeds,paired_per_seed}.csv` |
| Supplementary Table S6 | All comparison rankings | 21, 27 | `R4/transductive_main/masked/<dataset>/{paired_differences,new_comparators_paired_differences}.csv`; `R4/transductive_comparators/{absolute,paired_differences}.csv` |
| Supplementary Table S7 | Teacher and classifier ablations | 21 | `R4/transductive_main/item3_evaluation/<dataset>/paired_differences.csv` |
| Supplementary Table S8 | Selector feature groups | 21 | `R4/transductive_main/item3_evaluation/feature_pr_auc_exact.csv` |
| Supplementary Table S9 | Inductive settings | 20 | `R3/value_v2_ablations/ablations/evaluation/paired_differences.csv` |
| Supplementary Table S10 | Alternative training targets | 20, 25 | `R3/selector_v2/test/evaluation/{masked,thinned}/paired_differences.csv and references/{knockdown_summary,sex_paired,pdl1_paired}.csv; R4/dropout_posterior/evaluation/{masked,thinning,known,sex,protein}/ endpoint results.csv files` |
| Supplementary Table S11 | Binomial thinning | 21 | `R4/transductive_main/thinning_evaluation/{absolute,paired_differences}.csv` |
| Supplementary Table S12 | Recall by count | 8.3 | `R2/thinning_transfer/count_stratified_recall/count_stratified_recall.csv` |
| Supplementary Table S13 | Inserted-value accuracy | 9, 30 | `R4/current_tables/current/masked/{error_removed,log_error}.csv` |
| Supplementary Table S14 | Value accuracy by budget and stratum | 9, 30 | `R4/current_tables/current/masked/{error_removed,log_error_strata}.csv` |
| Supplementary Table S15 | Values at recorded zeros | 30 | `R4/current_tables/current/recorded/recorded_zero_fills.csv` |
| Supplementary Table S16 | Thinning value bias | 30 | `R4/current_tables/current/thinning/bias.csv` |
| Supplementary Table S17 | Full known-zero AUROCs and 95% intervals, including scRecover and scGPT | 22, 23, 28, 29, 38, 44 | `R4/round2_extras/part_a/table2_aurocs.csv`; `R4/table2_added/s17_rows.csv` |
| Supplementary Table S18 | Perturbation zero controls | 22, 28 | `R4/label_baselines/screens/verification_transductive/{auroc,effects}.csv`; `screens/verification_inductive_comparators/knockdown/{auroc,effects}.csv`; `screens/s16_paper_cells.csv` |
| Supplementary Table S19 | Masked F1 per screen | 22 | `R4/transductive_references/knockdown/masked_f1/masked_f1_by_screen.json` |
| Supplementary Table S20 | Rankings that use the same labels | 28, 29, 38 | `R4/main_tables_v2/estimates.csv` (`Supplementary` rows); Section S6 text: `R4/label_baselines/masked/<dataset>/report.json` and `R4/label_baselines/masked/supplementary/<dataset>/report.json` |
| Supplementary Table S21 | Sex-linked zero controls | 22, 23, 28 | `R4/sex_zero_comparators/reproduction_run/evaluation/<tissue>/auroc.csv`; sensitivity text: `R4/label_baselines/sex/colon/{agreed25,metadata34}_auroc.csv` |
| Supplementary Table S22 | Protein agreement | 22 | `R4/transductive_references/protein_cs/{evaluation_cd274/replicate_association,evaluation_cd274_raw_counts/continuous_protein_association,evaluation/continuous_protein_association}.csv` |
| Supplementary Table S23 | Protein associations, adjusted correlations and paired differences for all ten published rankings | 37, 42, 43 | `R4/published_protein/{association,paired_differences,s23_rows,paired_rows}.csv` |
| Supplementary Section S7, Full screen | PD-L1 across all 20,156 Papalexi cells | 22.1 | `R4/protein_followup/full/evaluation/{association,paired_differences,pdl1_pooled_rankings,global_fill}.csv` |
| Supplementary Table S24 | Masked downstream endpoints | 23 | `R4/transductive_downstream/summary/{s17_absolute,endpoint_changes,decomposition_counts}.csv` |
| Supplementary Table S25 | Gap closed at 1%, 10% and 100%, all cells used for fitting | 23, 31, 32, 40, 42 | `R4/published_downstream/s25_rows.tex` from both published downstream long tables |
| Supplementary Table S26 | Recorded-count downstream endpoints | 23 | `R4/transductive_downstream/summary/endpoint_changes.csv` |
| Supplementary Table S27 | Disease effects and null tests | 23 | `R4/transductive_downstream/summary/disease_all.csv` |
| Supplementary Table S28 | Reference mapping changes | 23 | `R4/transductive_downstream/disease/<tissue>_<fold>/deployment/annotation/<tissue>/overall.csv` |
| Supplementary Table S29 | Correlation null | 23 | `R4/transductive_downstream/correlation/null/counts/{summary,versus_reference}.csv` |
| Supplementary Table S30 | Fill-fraction rule | 30 | `R4/current_tables/current/rule/summary.csv`; `current/rule/{masked,recorded}/<unit>/calibration_report.json` |
| Supplementary Table S31 | Runtime | 18 | `runtime/runtime_table.csv` |
| Supplementary Table S32 | Scaling study | 26, 31, 33, 34 | `R3/scale/results/{paired_differences,resources,safe_fusion_totals}.csv`; Section S8 extensions: `R4/reviewer_extras/C_scale/results/paired_differences.csv`, `R4/scale_caps/genes_<2000 or 5000>/evaluation/results/paired_differences.csv`, `R4/scale_comparators/{paired_differences,resources}.csv` |
| Supplementary Table S33 | Lupus Treg example | 24, 35 | `R4/sle_donor_labels/comparisons/masked_f1_intervals.csv`; `comparisons/{q1_zeros_sex/per_gene,q3_clustering/treg_calls_summary,q3_clustering/clustering_mean_over_folds,focus_fills/focus_fill_counts,deploy_main/q5_modules,deploy_main/q4_effects,deploy_main/q4_null_de,q7_protein/fcrl3_protein_agreement}.csv` |
| Supplementary Section S10 / Table S34 | 41 blood RNA-protein pairs and library-size analysis | 36 | `R4/protein_multi/corrected/evaluation/{per_pair,summary}.csv`; `R4/protein_depth/tables/{association,differences,diagnostics,summary,ranking_summary}.csv`; `R4/protein_depth/tables/hao_pairs_table.tex` |

| Section | Source entry points | Output root under `artifacts/paper_evidence/` |
|---|---|---|
| 15 | `scripts/slurm_standard_imputers.sh`, `scripts/evaluate_fill_decisions.py` | `fill_decisions/table_s1.csv` |
| 19 | `scripts/slurm_colon_crossfit.sh`, `scripts/slurm_leakage_free_rerun.sh`, `scripts/slurm_r3_comparators.sh`, `scripts/slurm_r3_comparators_evaluate.sh`, `scripts/summarize_benchmark_data.py` | `review_round3/colon_crossfit/`, `review_round2/leakage_free/`, `review_round3/comparators/` |
| 20 | `scripts/slurm_v2_ablation.sh`, `scripts/slurm_v2_selector.sh`, `scripts/slurm_v2_selector_test.sh`, `scripts/slurm_v2_value.sh` | `review_round3/value_v2_ablations/`, `review_round3/selector_v2/` |
| 21 | `scripts/analyses/transductive_main/run.sh`, `scripts/slurm_thinning_transductive.sh`, `scripts/build_transductive_fillrate_summary.py`, `scripts/plot_overview_f1.py` | `review_round4/transductive_main/` |
| 22 | `scripts/analyses/transductive_references/reproduce.sh` | `review_round4/transductive_references/` |
| 22.1 | `scripts/analyses/protein_full/submit.sh`, `scripts/analyses/protein_full/evaluate.py` | `review_round4/protein_full/full/`, `review_round4/protein_followup/full/` |
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
| 36 | `scripts/analyses/protein_multi/reproduce.sh`, `scripts/analyses/protein_depth/reproduce.sh`, `scripts/table_hao_protein_pairs.py` | `review_round4/protein_multi/corrected/`, `review_round4/protein_depth/` |
| 37 | `scripts/analyses/protein_s20_rows/run.sh`, `scripts/analyses/protein_s20_rows/report.py` | `review_round4/protein_s20_rows/` |
| 38 | `scripts/analyses/main_tables_v2/run.sh`, `scripts/analyses/main_tables_v3/run.sh` | `review_round4/main_tables_v2/`, `review_round4/main_tables_v3/` |
| 39 | `scripts/analyses/protein_main_table/run.sh`, `scripts/analyses/protein_main_table/se.py` | `review_round4/protein_main_table/` |
| 40 | `scripts/analyses/fill_grid_dense/submit.sh` | `review_round4/fill_grid_dense/` (prerequisite) |
| 41 | `scripts/plot_method_schematic.py`, `scripts/plot_overview_f1.py` | `figures/safe_fusion_method.{pdf,png,svg}`, `figures/overview_f1_fillrate.{pdf,png}` |
| 42 | `scripts/analyses/published_downstream/prepare.py`, `scripts/analyses/published_downstream/submit.py`, `scripts/analyses/published_downstream/build_s25.py`, `scripts/plot_fill_grid_main.py` | `review_round4/published_downstream/`, `review_round4/published_downstream_allcells/`, `figures/fill_grid_main.{pdf,png}` |
| 43 | `scripts/analyses/published_protein/evaluate.py`, `scripts/analyses/published_protein/summarize.py` (submitted by section 42) | `review_round4/published_protein/` |
| 44 | `scripts/analyses/table2_added/prepare.py`, `scripts/analyses/table2_added/submit.sh`, `scripts/analyses/table2_added/report.py` | `review_round4/table2_added/` |

Section S8's calibration text reads `R4/reviewer_extras/A_calibration/calibration.csv`
(section 31). Its 5,000-gene, training-cap and additional-comparator text uses
sections 31, 33 and 34. Section S9 and S33 include both donor-label values.
Supplementary Figure S1 uses an 8.0 by 1.75 inch layout and the label
“CD274 RNA zeros selected (%)”.

Table S3 records settings rather than numerical estimates. Historical runtime
values require the original accounting; new runs measure their own costs.
The historical print audits require external manuscript inputs listed in
section 27. They are not bundled or silently replaced by the current paper.
The unavailable Dixit S15 inputs remain explicit. Table S10's omitted inductive Norman thinning posterior is not needed
for its reported endpoints.

Main Tables 1 and 2 use three blocks: Safe Fusion, published methods, and the
SVD and weighted kNN teachers as an ablation. Section 38 reuses v2 estimates and
bootstrap draws before adding v3 rows. S4 retains the full paired masked
comparisons; S17 retains known-zero intervals; S20 compares label-aware rankings.
Main Table 3 and S23 use section 43's all-method protein summaries. Nine
published methods provide ten rankings because scVI contributes both its mean
and its nonzero probability. The benchmark fits 3,400 cells and evaluates 655
CD274 zeros; the full screen fits 20,156 cells and evaluates 4,083 zeros.
scImpute exceeded 96 GB after 7.7 h; scRecover and EnImpute were stopped after
11.8 and 11.6 h. Their full-screen entries are not finished (n.f.).

Section 44 adds scRecover and scGPT to Table 2 and S17, using the exact current
known-zero inputs. Table 2 reports respectively 0.43 ± 0.03, 0.48 ± 0.01,
0.49 ± 0.05, 0.42 ± 0.03 and 0.38 ± 0.03, 0.60 ± 0.02, 0.20 ± 0.03,
0.27 ± 0.02 in CRISPRi, CRISPRa, pancreas and colon order.

Figure 2 uses a 12 by 3.4 inch canvas with one row of masked F1 curves.
Figure 3 uses a 12 by 6.6 inch canvas and 19 fractions from 0.25% through 100%.
It includes eight published count-valued methods plus detection-weighted
Safe Fusion; scGPT has no count-scale reconstruction. Every fit uses all
cells, including the SVD and ALRA refits. Norman response edges occupy the
upper-right panel; the lower-right panel holds the legend. Lines below -150%
are cut at the axis. The Safe Fusion band uses paired bootstrap intervals.
S25 reports the same gap percentages at 1%, 10% and 100%, with fused-value,
SVD and single-method-selector ablations. Its only final row builder is
`scripts/analyses/published_downstream/build_s25.py`; the earlier error-reduction
summaries are prerequisites. The original seven- and thirteen-fraction gates
compare generated reference values and formatted interval stars. Zebrafish
stage-error parity allows at most 5e-8, with identical printed values.
