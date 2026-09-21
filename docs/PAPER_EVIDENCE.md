# Safe Fusion results and limitations

## Summary

The paper uses an MLP selector with `apply_topk`. This document reports its
results alongside dense-output, logistic, and variance-gate ablations.
Matched-budget attribution with the logistic selector shows that the broad
effect is not generally caused by teacher fusion: fusion adds 1.50 percentage points of recovery in the
24-donor pancreas cross-fit, is slightly worse than no-fusion selection in
colon, and is effectively tied in CRISPRa, where max-teacher fill values are
better. The results support leakage-safe selective filling and component
attribution. They do not support broad Safe Fusion superiority. No selector
improves every biological endpoint. Logistic fixed-threshold selection is an
ablation.

## Benchmark design

For each dataset, 10% of observed nonzero RNA entries were masked before any
teacher, PCA, graph, or fusion model was fit. Models saw development and
validation donors only. The held-out test donors were used once for evaluation.
The baseline was chosen by minimum validation log1p MAE from raw counts, gene
mean, gene median, random filling, leakage-safe SVD, and leakage-safe kNN graph
smoothing. The test set did not choose the baseline.

| Dataset | Biological design | Matrix used | Locked test |
|---|---|---:|---:|
| Human colon epithelium, DOI `10.1016/j.immuni.2023.01.002` | 34 donors; epithelial identities and inflammation state | 5,077 cells × 1,211 genes | 9 donors |
| Human pancreatic islets, DOI `10.1038/s42255-022-00531-x` | 24 donors; 14 endocrine/exocrine/stromal cell labels and Control/AAB/T1D state | 3,418 cells × 1,208 genes | 6 donors |
| PBMC CITE-seq, GSE100501 | 3 donors; matched RNA and 83 antibody reagents | 2,500 cells × 1,062 genes | donor 3, descriptive only |

The PBMC antibody table contains 207,500 unique
`(cell_id, reagent_id, gene_id)` records. Reagents targeting the same gene were
not collapsed; for example, the four PTPRC reagents remain four distinct
measurements. Protein thresholds and protein-proxy labels were fit on donor 1
only.

## Dense-fusion reconstruction result

The primary metric is log1p MAE on the known positive counts hidden by the 10%
mask. Lower is better. Confidence intervals are paired cluster bootstraps over
held-out donors with 2,000 replicates.

| Dataset | Validation-selected baseline | Baseline MAE | Safe Fusion MAE | Relative improvement | Paired 95% CI |
|---|---|---:|---:|---:|---:|
| Colon | kNN graph smoothing | 0.4714 | 0.3987 | 15.4% | 7.6% to 24.3% |
| Pancreatic islets | kNN graph smoothing | 0.6133 | 0.4843 | 21.0% | 12.5% to 29.5% |
| PBMC CITE-seq | kNN graph smoothing | 0.4325 | 0.3600 | 16.8% | descriptive; one test donor |

Safe Fusion therefore passes the narrow reconstruction criterion on two
independent, donor-resolved datasets. The effect is not a cell-level
pseudoreplication result: both inferential confidence intervals resample whole
donors.

Secondary continuous metrics are mixed:

| Dataset | Method | Poisson deviance ↓ | Entry Spearman ↑ | Gene-wise Spearman ↑ | Cell-wise Spearman ↑ |
|---|---|---:|---:|---:|---:|
| Colon | graph | 5.771 | 0.7470 | 0.5792 | 0.6792 |
| Colon | Safe Fusion | 10.522 | 0.7538 | 0.5881 | 0.6530 |
| Pancreas | graph | 25.983 | 0.7229 | 0.4842 | 0.7347 |
| Pancreas | Safe Fusion | 57.722 | 0.7669 | 0.5322 | 0.6959 |
| PBMC | graph | 3.608 | 0.7790 | 0.3690 | 0.7592 |
| PBMC | Safe Fusion | 3.290 | 0.7796 | 0.3952 | 0.7449 |

Safe Fusion improves entry- and gene-wise rank correlation, but count deviance
is 82% worse than graph smoothing in colon and 122% worse in pancreas. This
fails the predeclared “no material deviance degradation” condition.

## Logistic fixed-threshold ablation

The logistic ablation ranks zeros from fused and teacher summaries, gene
expression and dropout context, and cell library size. Logistic regression is
trained on known masked positives in validation or development cells and
applied to held-out cells. Its threshold is fixed on the calibration split.
The separate variance-gate ablation has 0.4% recall at a 2% fill budget on
pancreatic-islet validation cells.

### Pancreatic islets (24-donor cross-fit, donor-level intervals)

| Metric | Raw | Calibrated 2% | Calibrated 5% | Calibrated 8.1% | Graph |
|---|---:|---:|---:|---:|---:|
| Masked-marker log1p MAE | 2.380 [2.304, 2.466] | 1.452 [1.401, 1.508] | 1.234 [1.182, 1.285] | 1.155 [1.107, 1.203] | 0.850 [0.812, 0.894] |
| Cell-type macro-F1 | 0.601 [0.567, 0.633] | 0.588 [0.552, 0.620] | 0.569 [0.530, 0.606] | 0.557 [0.517, 0.595] | 0.543 [0.513, 0.571] |
| Canonical marker PR-AUC | 0.512 [0.496, 0.527] | 0.532 [0.516, 0.548] | 0.540 [0.524, 0.556] | 0.545 [0.529, 0.562] | 0.702 [0.680, 0.723] |
| Off-target marker fill | 0.0% | 1.95% [1.45%, 2.52%] | 3.95% [3.01%, 5.04%] | 5.87% [4.61%, 7.22%] | 17.9% [16.0%, 19.7%] |
| T1D logFC Spearman | 0.948 [0.929, 0.957] | 0.949 [0.931, 0.958] | 0.941 [0.922, 0.948] | 0.930 [0.907, 0.937] | 0.567 [0.300, 0.596] |
| AAB logFC Spearman | 0.907 [0.898, 0.929] | 0.917 [0.906, 0.935] | 0.904 [0.894, 0.924] | 0.873 [0.860, 0.906] | 0.269 [0.107, 0.440] |

Paired donor differences at 2%: masked-marker MAE −0.928, cell identity
−0.013 (95% CI −0.026 to +0.002), canonical marker PR-AUC +0.020 (CI +0.018
to +0.022), and ectopic fill +1.95 pp (CI +1.45 to +2.52). T1D and AAB logFC
Spearman remain near raw.

### Colon (9 locked test donors)

| Metric | Raw | Calibrated 2% | Calibrated 5% | Calibrated 8.1% | Graph |
|---|---:|---:|---:|---:|---:|
| Masked-marker log1p MAE | 1.646 [1.441, 1.843] | 1.138 [1.039, 1.256] | 0.989 [0.883, 1.115] | 0.918 [0.829, 1.022] | 0.633 [0.560, 0.707] |
| Cell-type macro-F1 | 0.477 [0.407, 0.539] | 0.461 [0.383, 0.533] | 0.442 [0.372, 0.504] | 0.429 [0.353, 0.493] | 0.489 [0.421, 0.563] |
| Canonical marker PR-AUC | 0.339 [0.312, 0.366] | 0.352 [0.319, 0.386] | 0.360 [0.325, 0.396] | 0.366 [0.333, 0.401] | 0.495 [0.457, 0.534] |
| Off-target marker fill | 0.0% | 0.50% [0.31%, 0.72%] | 1.20% [0.90%, 1.55%] | 2.03% [1.56%, 2.58%] | 12.8% [11.5%, 14.1%] |

Paired donor differences at 2%: masked-marker MAE −0.508 (CI −0.633 to
−0.372), cell identity −0.016 (CI −0.034 to +0.002, non-inferior), canonical
marker PR-AUC +0.014 (CI +0.006 to +0.023), ectopic fill +0.50 pp (CI +0.31 to
+0.72).

### CRISPRa perturbation screen (64 effective single-gene conditions)

| Metric | Raw | Calibrated 2% | Calibrated 5% | Graph |
|---|---:|---:|---:|---:|
| Masked-entry log1p MAE | 1.325 | 0.793 | 0.594 | 0.425 |
| Target direction preserved | 0.922 [0.844, 0.984] | 0.906 [0.813, 0.969] | 0.922 [0.844, 0.984] | 0.625 [0.516, 0.750] |
| Signature logFC Spearman | 0.879 [0.876, 0.883] | 0.886 [0.882, 0.889] | 0.872 [0.869, 0.876] | 0.146 [0.133, 0.159] |
| Ectopic target fill in controls | 0.0% | 1.6% [0.0%, 4.5%] | 10.8% [4.4%, 18.4%] | 21.7% [13.4%, 30.5%] |
| Target recovery in activated cells | 0.958 | 0.970 [0.946, 0.989] | 0.977 [0.955, 0.996] | 0.363 |

At 2% the calibrated selector recovers 40% of the masked-entry error (MAE
1.325 → 0.793) while preserving target directions (0.906 vs raw 0.922; paired
difference −0.016, CI −0.047 to 0.000), improving target recovery (+0.011, CI
+0.002 to +0.022), keeping ectopic target fill at 1.6%, and adding no false
responses in controls.

### Interpretation and component attribution

The calibrated rule recovers masked signal and largely preserves the measured
biological endpoints. Exact matched-coverage attribution is essential:

| Dataset | Full recovery | No-fusion recovery | Full minus no-fusion (95% CI) |
|---|---:|---:|---:|
| Pancreas | 39.75% | 38.25% | +1.50 pp [1.09, 1.88] |
| Colon | 39.30% | 39.58% | -0.28 pp [-0.43, -0.03] |
| CRISPRa | 56.01% | 55.95% | +0.06 pp [0.01, 0.11] |

Fusion therefore adds a small, reliable increment only in pancreas. It is not
the general cause of the selective-filling benefit. The method also remains
behind graph smoothing or ALRA on several reconstruction endpoints. The
defensible contribution is the calibrated decision rule, leakage-safe
evaluation, and negative/qualified attribution result.

### MLP exact-budget selection and selector ablations

The architecture comparison includes logistic regression, histogram gradient
boosting, extra trees, a three-hidden-layer MLP, and a four-model percentile-rank ensemble.
All models used the same eight features, the same 400,000-row stratified fit
cap, the same development/test partitions, identical fused fill values, and
exactly matched test coverage without test labels.

| Selector | Pancreas recovery | Colon recovery | CRISPRa recovery |
|---|---:|---:|---:|
| Logistic | 39.98% | 39.33% | 55.85% |
| Histogram GBDT | 42.58% | 39.87% | 56.97% |
| Extra trees | 42.67% | 39.51% | 56.83% |
| MLP | 42.81% | 40.22% | 57.00% |
| Rank ensemble | 42.84% | 40.19% | 56.96% |

At 2%, MLP-minus-logistic recovery is +2.84 pp in pancreas (donor-bootstrap
CI +2.05 to +3.50), +0.90 pp in colon (-0.29 to +1.90), and +1.15 pp in
CRISPRa (+0.98 to +1.34). In the direct FT-Transformer sensitivity,
FT-versus-MLP recovery is 42.48% versus 42.85% in pancreas (difference -0.37
pp, CI -0.63 to -0.12), 39.95% versus 39.93% in colon (+0.03 pp, CI -0.96 to
+0.59), and 56.944% versus 56.942% in CRISPRa (+0.002 pp, CI -0.077 to
+0.093). Thus it is worse in pancreas and tied in colon and CRISPRa. Its
CRISPRa CPU fit took 1,181 s versus 15.8 s for MLP (about 75-fold longer), so
attention is not the source of an additional gain and does not justify the
extra dependency or compute.

The calibrator exposes two operating modes. The logistic ablation's
`calibration_threshold` mode transfers a development cutoff and keeps
`test_values_used_for_thresholds: false`; realized test fill can drift, most
strongly in CRISPRa, where a nominal 2% development threshold fills about
0.4% of test zeros. The manuscript `apply_topk` mode uses no test labels but
does use the unlabelled target score quantile to deliver exact requested
coverage. With up to two million fit candidates:

| Exact target budget | Pancreas logistic / MLP | Colon logistic / MLP | CRISPRa logistic / MLP |
|---|---:|---:|---:|
| 2% | 39.75% / 43.02% | 39.30% / 40.29% | 56.00% / 57.02% |
| 5% | 57.63% / 59.33% | 53.06% / 53.74% | 64.76% / 65.11% |
| 10% | 65.18% / 66.43% | 60.54% / 61.01% | 69.49% / 69.57% |

The MLP gain is largest in the sparse 2% regime, but biological effects are
mixed. In pancreas at exact 2%, MLP versus logistic improves detailed
cell-identity macro-F1 by +0.0128 (CI +0.0086 to +0.0167), development-marker
PR-AUC by +0.00106 (+0.00058 to +0.00155), and ectopic marker fill by -1.70
pp (-2.12 to -1.29). It also improves T1D/AAB disease-logFC RMSE by
-0.0259/-0.0093 and Spearman by +0.0028/+0.0040; all four intervals exclude
zero. However, canonical-marker PR-AUC falls by 0.0067 and masked
canonical-marker MAE worsens by 0.372. Colon shows fewer ectopic fills and a
small development-marker gain but worse masked canonical-marker MAE. CRISPRa
response PR-AUC improves, while signature Spearman and top-decile direction
worsen significantly; 5-10% budgets are not biologically safe for that screen.

MLP exact-budget selection is the manuscript method. Its reconstruction gains
and pancreas biological positives do not establish universal biological
improvement. The Perturb-seq tradeoffs do not support increasing its fill
budget.

### Fill-rate comparison (fraction of locked-test zeros that become nonzero)

| Method | Pancreatic islets | Colon | CRISPRa |
|---|---:|---:|---:|
| Raw | 0.0% | 0.0% | 0.0% |
| Gene mean / median / random fill | 100% | 100% | — |
| Leakage-safe SVD | 73% | 83% | — |
| kNN graph smoothing | 62% | 77% | 19% |
| Official ALRA (zero-preserving) | 13.7% | 18.6% | 12.2% |
| SAVER (uncertainty-aware) | 32.5% | 45.1% | 10.8%¹ |
| scGCL architecture sensitivity | 50.3% | 50.7% | —² |
| Dense Safe Fusion | 100% | 100% | 100% |
| Calibrated 2% | 2.1% | 2.5% | 0.4% |
| Calibrated 5% | 5.4% | 6.0% | 1.9% |
| Calibrated 8.1% | 9.2% | 9.8% | — |
| Calibrated 10% | 11.6% | 12.1% | — |

Classical imputers and dense Safe Fusion fill essentially every observed zero;
calibrated Safe Fusion fills only the budgeted fraction and still recovers
39–56% of masked error at exact matched 2% coverage. The official ALRA zero-preserving baseline
(KlugerLab algorithm, fit on training cells with test projection) reaches
near-graph masked-entry MAE while filling only 12–19% of zeros: pancreas
0.576, colon 0.689, CRISPRa 0.485 (graph: 0.850/0.633/0.425). ALRA is
therefore a strong zero-preserving comparator and supports the paper's
central point that reconstruction quality and fill aggressiveness are
decoupled.

The pinned scGCL default is not numerically portable to these matrices:
`lr=1e-3` becomes non-finite at epoch 68 on pancreas, `1e-4` at epoch 165,
and `1e-5` after epoch 240 on colon. A common 300-epoch `1e-6` architecture
sensitivity completes with masked-entry MAE 1.504/1.270 versus raw
1.784/1.532 (pancreas/colon), but fills 50.3%/50.7% of test zeros. This is a
qualified transductive sensitivity, not untouched standard usage or evidence
for selective biological recovery.

¹ SAVER uses its standard full-matrix usage (no held-out projection API) and
the CRISPRa column is on the 1,500-gene subset (full-matrix SAVER exceeds the
cluster time limit). SAVER masked-entry MAE: pancreas 1.100, colon 0.973,
CRISPRa-subset 0.707 (same-subset raw 1.331 / graph 0.423 / ALRA 0.483).

² scGCL uses the pinned official architecture and preprocessing but requires
the disclosed `1e-6` stability learning rate; like SAVER it is transductive
and has no held-out projection API. CRISPRa was not run for this sensitivity.

## Post-facto marker replication

Markers were discovered from each method in one donor fold and validated
against untouched reference counts in the opposite fold, then the folds were
swapped. Replication precision@k is 0.5668 for calibrated 2% versus 0.5667 for
raw in pancreas (difference 0.0001, 95% CI -0.0007 to 0.0009) and 0.4381
versus 0.4340 in colon (difference 0.0042, CI 0.0004 to 0.0078). Graph
smoothing is modestly better in both; dense fusion is significantly worse.
This is neutral-to-tiny-positive evidence, not a large marker-discovery gain.

## Calibrated PBMC protein check

In the locked PBMC donor, median RNA--protein Spearman is 0.1183 for raw and
calibrated 2%/5%; RNA/protein kNN overlap changes from 0.0909 to 0.1000/0.1024.
Dense fusion lowers these endpoints to 0.0843/0.0680. The sparse outputs do not
preferentially fill protein-high RNA zeros, so the result supports preservation
but not protein-certified dropout identification.

## Biological evidence

*This section and the PBMC/CRISPRa sections below describe dense-output
ablations. Interpret them separately from MLP `apply_topk` results above.*

### Cell identity, markers, and disease effects

All metrics below use held-out donors. Marker PR-AUC evaluates recovery of
development-defined cell-type markers; DE is donor-by-condition pseudobulk
where the test cohort supports it.

| Dataset / metric | Raw | Graph baseline | Safe Fusion |
|---|---:|---:|---:|
| Colon donor-held-out cell-type macro-F1 | 0.538 | 0.542 | 0.321 |
| Colon marker PR-AUC | 0.761 | 0.832 | 0.675 |
| Colon marker rank Spearman | 0.845 | 0.875 | 0.766 |
| Colon inflammation logFC Spearman¹ | 0.967 | 0.590 | 0.336 |
| Colon inflammation logFC RMSE¹ | 0.258 | 1.098 | 1.369 |
| Pancreas donor-held-out cell-type macro-F1 | 0.605 | 0.573 | 0.443 |
| Pancreas marker PR-AUC | 0.754 | 0.859 | 0.641 |
| Pancreas marker rank Spearman | 0.863 | 0.928 | 0.727 |

Dense Safe Fusion does not preserve cell identity or marker biology. The loss
replicates in immune-independent colon and pancreatic tissues.

¹ Colon DE uses only the one locked test donor (117351) with paired inflamed and
uninvolved biosamples; it is descriptive, not inferential. The donor-level
inferential DE replication is the pancreatic-islet T1D/AAB contrast below.

### Donor cross-fit confirmation (pancreatic islets)

To remove any remaining pseudoreplication concern, every pancreatic-islet donor
was evaluated out of fold in a three-fold, condition-stratified cross-fit. Each
method was refit three times, always without the held-out donors; the
assembled out-of-fold predictions cover all 24 donors exactly once. Confidence
intervals are 2,000-replicate donor-stratified bootstraps. All method contracts
passed the leakage audit (`test_used_for_fit: false`, thresholds fit on
development donors only).

| Metric | Raw | Graph baseline | Dense Safe Fusion |
|---|---:|---:|---:|
| Detailed cell-type macro-F1 | 0.601 [0.567, 0.633] | 0.543 [0.513, 0.571] | 0.457 [0.425, 0.487] |
| Broad cell identity macro-F1 | 0.730 [0.688, 0.766] | 0.697 [0.658, 0.733] | 0.561 [0.524, 0.596] |
| Canonical marker PR-AUC | 0.512 [0.496, 0.527] | 0.702 [0.680, 0.723] | 0.590 [0.572, 0.606] |
| Development marker PR-AUC | 0.563 [0.543, 0.582] | 0.738 [0.721, 0.754] | 0.558 [0.539, 0.576] |
| Development marker rank Spearman | 0.687 [0.658, 0.712] | 0.825 [0.802, 0.845] | 0.652 [0.627, 0.676] |
| Off-target marker zero fill | 0.0% | 17.9% [16.0%, 19.7%] | 98.7% [98.4%, 98.9%] |
| Masked canonical-marker log1p MAE | 2.380 [2.304, 2.466] | 0.850 [0.812, 0.894] | 0.937 [0.894, 0.976] |

The paired donor-level differences are decisive: dense Safe Fusion is
significantly worse than graph smoothing on masked-marker recovery
(+0.087, 95% CI 0.042–0.134), cell identity (−0.086, CI −0.115 to −0.054),
off-target fill (+0.808, CI 0.790–0.826), and T1D disease-logFC Spearman
(−0.191, CI −0.266 to −0.101).

T1D-versus-control pseudobulk effects (5 T1D vs 11 control donors, 14 cell
types):

| Metric | Raw | Graph baseline | Dense Safe Fusion |
|---|---:|---:|---:|
| Disease logFC Spearman | 0.948 [0.929, 0.957] | 0.567 [0.300, 0.596] | 0.376 [0.135, 0.372] |
| Disease logFC RMSE | 0.247 | 0.692 | 0.818 |
| Direction agreement, top decile | 0.996 | 0.899 | 0.742 |

The same pattern holds for AAB-versus-control (8 vs 11 donors): logFC Spearman
0.907 raw, 0.269 graph, 0.138 dense Safe Fusion. The biological failure of the
dense output is therefore donor-level real, not a split artifact.

### Colon donor-level confirmation (locked test donors)

The colon biology was recomputed per donor on the 9 locked test donors using
the same frozen method contracts, with 2,000-replicate donor bootstraps. The
marker panel is the prespecified epithelial/inflammation panel; condition
markers are excluded here because only one test donor has inflamed tissue.
Donor-averaged values are more conservative than the pooled test-set point
estimates above but reproduce the same direction with intervals.

| Metric | Raw | Graph baseline | Dense Safe Fusion |
|---|---:|---:|---:|
| Detailed cell-type macro-F1 | 0.477 [0.407, 0.539] | 0.489 [0.421, 0.563] | 0.271 [0.204, 0.333] |
| Broad cell identity macro-F1 | 0.515 [0.440, 0.589] | 0.592 [0.507, 0.677] | 0.251 [0.180, 0.318] |
| Canonical marker PR-AUC | 0.339 [0.312, 0.366] | 0.495 [0.457, 0.534] | 0.305 [0.264, 0.348] |
| Canonical marker effect Spearman | 0.966 [0.935, 0.992] | 0.741 [0.655, 0.817] | 0.574 [0.354, 0.714] |
| Development marker PR-AUC | 0.484 [0.448, 0.516] | 0.621 [0.578, 0.662] | 0.490 [0.426, 0.542] |
| Development marker rank Spearman | 0.572 [0.507, 0.626] | 0.687 [0.638, 0.734] | 0.546 [0.453, 0.621] |
| Off-target marker zero fill | 0.0% | 12.8% [11.5%, 14.1%] | 96.6% [96.0%, 97.5%] |
| Masked canonical-marker log1p MAE | 1.646 [1.441, 1.843] | 0.633 [0.560, 0.707] | 0.616 [0.519, 0.726] |

The paired donor differences are decisive on identity and specificity: dense
Safe Fusion is significantly worse than graph smoothing on cell-type macro-F1
(−0.218, 95% CI −0.263 to −0.174), marker PR-AUC (−0.189, CI −0.216 to −0.163),
and off-target fill (+0.838, CI 0.820–0.858). Masked-marker recovery is
statistically tied with graph smoothing (−0.017, CI −0.077 to 0.042), so the
reconstruction advantage does not translate into better marker recovery here.
The identity/marker/ectopic failure is therefore donor-level real in both
datasets; the inferential disease-DE failure is established in pancreatic
islets.

### Prespecified biological markers and ectopic expression

The colon panel contains epithelial identity and inflammatory genes. The
pancreatic panel contains 43 prespecified beta-, alpha-, delta-, PP-, epsilon-,
acinar-, ductal-, stellate-, endothelial-, immune-, and T1D-response markers.
The off-target statistic is the fraction of truly zero marker entries in the
wrong cell population that are imputed above 0.5 counts.

| Dataset / method | Masked-marker log1p MAE ↓ | Marker effect Spearman ↑ | Mean marker PR-AUC ↑ | Off-target zero fill ↓ |
|---|---:|---:|---:|---:|
| Colon graph | 0.672 | 0.766 | 0.367 | 16.9% |
| Colon Safe Fusion | 0.671 | 0.869 | 0.246 | 98.1% |
| Pancreas graph | 0.870 | 0.983 | 0.611 | 32.3% |
| Pancreas Safe Fusion | 0.939 | 0.938 | 0.466 | 99.5% |

Safe Fusion recovers some individual hidden markers, but it does so by filling
nearly every off-target marker zero. That is not biologically specific marker
recovery.

### Independent PBMC protein evidence

PBMC protein is an independent continuous proxy, not certified RNA absence.
All values use locked donor 3, with all duplicate antibody reagents retained.

| Method | Median RNA–protein Spearman ↑ | RNA/protein kNN overlap ↑ | Fill of RNA zeros with high protein | Fill of RNA zeros with low protein |
|---|---:|---:|---:|---:|
| Raw corrupted RNA | 0.118 | 0.091 | 0.0% | 0.0% |
| Graph smoothing | 0.127 | 0.078 | 0.0% | 0.0% |
| Safe Fusion | 0.084 | 0.068 | 99.6% | 99.4% |

Safe Fusion does not preferentially refill protein-supported RNA zeros: the
high-protein and low-protein groups are filled at essentially the same rate.
RNA–protein rank agreement and neighborhood agreement both decline.

## Perturbation preservation (CRISPRa, Norman et al. 2019)

The perturbation-direction gate was tested on the Norman et al. 2019 CRISPRa
Perturb-seq screen (GSE133344, A549), using the Gears-processed file with its
raw UMI counts layer. The locked benchmark contains 64 single-gene activation
conditions that passed target-effect QC (target log2FC ≥ 0.5, Mann–Whitney
p < 0.05 versus control), 60 cells per condition and 1,000 control cells,
5,045 genes. Ten percent of observed nonzero counts were masked before any
teacher or fusion model was fit; 70% of cells per condition were development,
30% were locked test. Units are perturbations because the processed file has no
replicate column; intervals are 2,000-replicate bootstraps over the 64
conditions. Response genes per condition are the top-decile reference logFC
genes, defined from development cells only; the control false-response rate is
the fraction of genes with |logFC| > 1 in a control-versus-control
pseudo-contrast on test cells.

Primary masked-entry recovery on locked test cells (log1p MAE, lower better):

| Method | Masked-entry log1p MAE |
|---|---:|
| Corrupted raw | 1.325 |
| Graph smoothing | 0.425 |
| Dense Safe Fusion | 0.368 (13.5% better than graph) |
| Selective 2% | 1.325 |

Perturbation-preservation metrics on locked test cells:

| Metric | Raw | Graph baseline | Dense Safe Fusion | Selective 2% |
|---|---:|---:|---:|---:|
| Target direction preserved | 0.922 [0.844, 0.984] | 0.625 [0.516, 0.750] | 0.563 [0.438, 0.688] | 0.922 [0.844, 0.984] |
| Target logFC absolute error | 0.038 | 0.101 | 0.110 | 0.039 |
| Signature logFC Spearman | 0.879 [0.876, 0.883] | 0.146 [0.133, 0.159] | 0.092 [0.079, 0.105] | 0.715 [0.711, 0.719] |
| Response-gene PR-AUC | 0.290 | 0.354 | 0.233 | 0.280 |
| Target recovery in activated cells | 0.958 | 0.363 | 0.877 | 0.958 |
| Ectopic target fill in controls | 0.0% | 21.7% [13.4%, 30.5%] | 90.6% [84.9%, 95.6%] | 0.0% |
| Control false-response rate | 0.04% | 0.0% | 0.0% | 0.04% |

Dense Safe Fusion again wins the masked-entry reconstruction task but fails
perturbation preservation: it preserves the direction of only 56% of effective
CRISPRa targets (raw: 92%), collapses the genome-wide response signature
(Spearman 0.09 vs raw 0.88), and induces ectopic target expression in 90.6% of
control-cell zeros. Graph smoothing also distorts directions (62.5%
preserved), so perturbation direction is not safe under either imputer. The 2%
guardrail preserves direction and specificity (ectopic fill 0%) but its
masked-entry MAE equals raw, so it again provides no recovery gain. The
reconstruction–biology failure therefore extends from immune and non-immune
tissue atlases to CRISPRa perturbation biology.

## Does selective filling solve the problem?

*This section describes the variance-based gate ablation. Compare it with MLP
exact-budget selection above.*

A frozen q=0.90 confidence gate with a 2% development/validation fill budget
and raw fallback was also tested. In pancreatic islets it preserved biology:
macro-F1 was 0.609 versus 0.605 for raw, marker PR-AUC was 0.752 versus 0.754,
and off-target marker filling fell to 1.45%. But masked-entry log1p MAE was
1.772 versus 0.613 for graph smoothing. The same pattern appeared in colon at
2%: macro-F1 0.517 versus 0.538 raw, marker PR-AUC 0.756 versus 0.761 raw, and
DE Spearman 0.943 versus 0.967 raw.

The same gate was re-tested inside the pancreatic-islet donor cross-fit, with
thresholds fit per fold on development donors only and raw fallback. All 24
donors are evaluated out of fold; intervals are donor-stratified 2,000-replicate
bootstraps.

| Method / fill budget | Cell-type macro-F1 | Canonical marker PR-AUC | Off-target fill | Masked-marker log1p MAE |
|---|---:|---:|---:|---:|
| Raw | 0.601 [0.567, 0.633] | 0.512 [0.496, 0.527] | 0.0% | 2.380 [2.304, 2.466] |
| Selective 2% | 0.594 [0.562, 0.622] | 0.512 [0.496, 0.527] | 1.24% [0.96%, 1.55%] | 2.377 [2.301, 2.462] |
| Selective 5% | 0.566 [0.540, 0.590] | 0.515 [0.500, 0.531] | 2.72% [2.23%, 3.27%] | 2.376 [2.299, 2.460] |
| Selective 8.1% | 0.552 [0.526, 0.576] | 0.518 [0.502, 0.534] | 4.35% [3.79%, 4.99%] | 2.374 [2.297, 2.458] |
| Selective 10% | 0.549 [0.521, 0.574] | 0.518 [0.503, 0.534] | 4.74% [4.10%, 5.44%] | 2.373 [2.296, 2.458] |
| Graph baseline | 0.543 [0.513, 0.571] | 0.702 [0.680, 0.723] | 17.9% | 0.850 [0.812, 0.894] |

At 2% the guardrail is statistically near raw on identity and marker
specificity: cell-type F1 difference −0.007 (95% CI −0.0145 to +0.0010),
canonical marker PR-AUC difference +0.0001 (CI −0.0014 to +0.0019), and
off-target fill 1.24% (0.96–1.55%). But masked-marker log1p MAE is 2.377,
indistinguishable from raw 2.380 and far worse than graph smoothing 0.850: the
guardrail fills so few zeros that it recovers almost none of the masked
positives. T1D logFC Spearman recovers from 0.376 (dense) to 0.827 (2%), but
that is still below raw 0.948. The tradeoff is therefore confirmed with
donor-level intervals: selective filling preserves most biology at the cost of
abandoning the reconstruction advantage.

The same frozen gate on the 9 locked colon donors (thresholds fit on
development/validation; raw fallback) reproduces the pattern with donor-level
intervals:

| Method / fill budget | Cell-type macro-F1 | Canonical marker PR-AUC | Off-target fill | Masked-marker log1p MAE |
|---|---:|---:|---:|---:|
| Raw | 0.477 [0.407, 0.539] | 0.339 [0.312, 0.366] | 0.0% | 1.646 [1.441, 1.843] |
| Selective 2% | 0.462 [0.382, 0.531] | 0.347 [0.322, 0.374] | 1.18% [0.75%, 1.67%] | 1.644 [1.439, 1.838] |
| Selective 5% | 0.445 [0.370, 0.511] | 0.349 [0.324, 0.376] | 3.23% [2.24%, 4.32%] | 1.643 [1.438, 1.837] |
| Selective 8.1% | 0.431 [0.374, 0.479] | 0.354 [0.328, 0.381] | 5.00% [3.53%, 6.69%] | 1.642 [1.438, 1.836] |
| Selective 10% | 0.425 [0.367, 0.474] | 0.358 [0.331, 0.385] | 5.73% [4.33%, 7.32%] | 1.641 [1.438, 1.833] |
| Graph baseline | 0.489 [0.421, 0.563] | 0.495 [0.457, 0.534] | 12.8% | 0.633 [0.560, 0.707] |

At 2% the colon guardrail is statistically near raw: cell-type F1 difference
−0.015 (95% CI −0.032 to +0.002), canonical marker PR-AUC +0.008 (CI +0.005 to
+0.012), off-target fill 1.18% (0.75–1.67%), and masked-marker MAE 1.644 versus
raw 1.646 — again no reconstruction gain. Both datasets therefore show the
same donor-level tradeoff.

The variance-based gate approximately preserves biology but does not retain the
reconstruction advantage.

## External CRISPRi, KO, and ECCITE-seq validation

Harmonized raw-count matrices were obtained from the scPerturb Zenodo record
for Adamson/Weissman CRISPRi (GSE90546), Dixit/Regev Cas9 KO (GSE90063),
and Papalexi/Satija ECCITE-seq (GSE153056).
Cells were split before target QC and variable-gene selection. The Adamson
benchmark retains 30 targets with development-only target log2FC <= -0.25 and
Mann-Whitney p < 0.05. The Dixit object contains ten well-powered assigned
targets; direct target RNA is not used as an efficacy gate because Cas9 can
disrupt protein without lowering its transcript. None of the three harmonized
objects has an independent replicate field, so target is the bootstrap unit.

At the calibrated 2% operating point, simulated-mask selector precision/recall
is 98.85%/8.15% in Adamson, 98.31%/11.94% in Dixit, and 96.44%/10.50% in
Papalexi. On held-out cells:

| Dataset / endpoint | Raw | Calibrated 2% | Difference vs raw (95% target-bootstrap CI) |
|---|---:|---:|---:|
| Adamson CRISPRi signature logFC Spearman | 0.8873 | 0.8970 | +0.00975 [0.00828, 0.01116] |
| Adamson direct target logFC absolute error | 0.0822 | 0.2202 | +0.1380 [0.0298, 0.2972], worse |
| Dixit KO signature logFC Spearman | 0.8596 | 0.8749 | +0.01527 [0.01345, 0.01691] |
| Dixit KO response PR-AUC | 0.10675 | 0.10692 | +0.00017 [-0.00039, 0.00074] |
| Papalexi ECCITE-seq signature logFC Spearman | 0.89371 | 0.90369 | +0.00998 [0.00850, 0.01144] |
| Papalexi target logFC absolute error | 0.04608 | 0.04154 | -0.00454 [-0.01307, 0.00169], neutral |

Dense fusion reduces signature Spearman to 0.403 in Adamson, 0.416 in Dixit,
and 0.654 in Papalexi and fills most evaluated ectopic target zeros. The external
screens therefore supply a real but narrow positive: selective filling
slightly improves genome-wide perturbation-signature preservation. They do
not support improved perturbation biology overall because the CRISPRi target
effect worsens and response-gene ranking is neutral.

## Real developmental trajectory validation

The CellRank zebrafish axial-mesoderm matrix contains 2,434 cells, 12 ordered
stages, and three lineages. The split is stratified within stage before
development-only variable-gene selection. Development-fitted kNN time and
lineage decoders are evaluated on 729 held-out cells; intervals resample the
12 stages. The processed object has no embryo/library replicate field.

At 2%, dynamic-gene curve Spearman improves from 0.99075 to 0.99444
(difference +0.00370, 95% CI 0.00224 to 0.00535) and stage-pseudobulk
Spearman from 0.99213 to 0.99247 (+0.00035, 0.00008 to 0.00068). Stage-rank
MAE changes from 0.35346 to 0.35214 (-0.00132, CI -0.01279 to 0.01270),
lineage accuracy from 0.96253 to 0.96691 (CI for the difference includes
zero), and adjacent-stage neighbor rate is neutral. Dense fusion significantly
degrades dynamic-gene correlation, pseudobulk correlation, neighborhood
ordering, and stage-rank error. Thus trajectory expression curves show small
positive preservation effects, but developmental ordering itself is not
improved.

## Complete five-task downstream evaluation on real data

The final selector was evaluated at every integer selected-zero fraction from
1 through 10%. Pancreas and colon analyses hold out donors. The zebrafish
analysis holds out cells within each of 12 developmental stages and fits one
common PCA and downstream model on the corrupted development cells. The GRN
analysis uses real genetic interventions rather than SERGIO or another
simulator. For each regulator, development cells define response edges whose
absolute log2 fold change is in the top 5 or 10% and whose sign agrees in two
development halves. Held-out intervention and control cells then evaluate edge
recovery. The source gene is excluded. These are causal intervention-response
edges and are not presented as direct physical binding interactions.

| Task and primary endpoint | Corrupted raw | Weighted kNN | Safe Fusion, 1--10% | Fractions better than raw | Fractions with interval excluding zero |
|---|---:|---:|---:|---:|---:|
| Colon clustering ARI | 0.1762 | 0.1891 | 0.1771--0.1915 | 10/10 | 8/10 |
| Pancreas clustering ARI | 0.3804 | 0.3804 | 0.3803--0.3838 | 9/10 | 0/10 |
| Colon canonical-marker PR AUC | 0.3387 | 0.4945 | 0.3488--0.3912 | 10/10 | 10/10 |
| Pancreas canonical-marker PR AUC | 0.5116 | 0.7024 | 0.5201--0.5613 | 10/10 | 10/10 |
| AAB versus control logFC Spearman | 0.9070 | 0.2690 | 0.8695--0.9252 | 6/10 | 5/10 |
| T1D versus control logFC Spearman | 0.9485 | 0.5674 | 0.9251--0.9529 | 4/10 | 3/10 |
| Zebrafish dynamic-gene Spearman | 0.9904 | 0.9687 | 0.9925--0.9951 | 10/10 | 10/10 |
| Adamson CRISPRi edge PR AUC | 0.2639 | 0.3000 | 0.2652--0.2707 | 10/10 | 10/10 |
| Dixit knockout edge PR AUC | 0.1099 | 0.1109 | 0.1100--0.1109 | 10/10 | 0/10 |
| Norman CRISPRa edge PR AUC | 0.3730 | 0.3945 | 0.3344--0.3667 | 0/10 | 0/10 |
| Papalexi ECCITE-seq edge PR AUC | 0.2697 | 0.3121 | 0.2703--0.2745 | 10/10 | 9/10 |

The clearest positive result is preservation of zebrafish dynamic expression
curves across the full range. Marker PR AUC also improves across all fractions
in both tissues, but weighted kNN remains higher and increasing the number of
selected zeros increases off-target marker filling. Colon clustering improves;
pancreas ARI is neutral although its neighborhood purity improves at all ten
fractions. Disease-effect preservation improves only in the lower part of the
range, showing why the selected fraction is a biological safety parameter.

The real interventional GRN result is mixed. Safe Fusion improves response-edge
PR AUC throughout the range in Adamson CRISPRi and Papalexi ECCITE-seq, is
numerically neutral in Dixit knockout, and worsens Norman CRISPRa edge ranking.
Weighted kNN remains stronger for Adamson and Papalexi edge ranking. Effect-size
correlation to the held-out unmasked counts nevertheless improves throughout
the range in all three loss-of-function screens. The perturbation objects do
not provide complete independent replicate fields, so the 128 perturbed
regulators, rather than cells, are the bootstrap units.

## What the experiments demonstrate

1. Dense output is not biologically safe (ectopic fill 96–99%,
   identity/marker/DE/perturbation distortion), and the variance gate
   recovered almost no masked positives (0.4% recall at 2%).
2. Logistic matched-coverage selection recovers 39–56% of masked error at
   exactly matched 2% coverage while largely preserving measured identity,
   marker, DE, protein, and CRISPRa endpoints; external CRISPRi/KO/ECCITE-seq signatures
   and zebrafish dynamic-gene curves show small additional gains.
3. Matched-budget ablation attributes only a small pancreas-specific increment
   to fusion; teacher/context selection explains the colon and CRISPRa effect.
4. The calibrated rule uses only validation/development masked positives for
   training. Fixed-threshold contracts also fit thresholds there; the separate
   MLP exact-budget method uses the unlabelled target score quantile but no
   target labels. All contracts pass the corresponding leakage audit.
5. Post-facto markers are neutral-to-tiny-positive. Trajectory expression
   curves improve across the full 1--10% range, and marker PR AUC improves in
   both tissues with off-target and baseline tradeoffs. Real intervention-edge
   recovery improves in Adamson CRISPRi and Papalexi ECCITE-seq, is neutral in
   Dixit knockout, and worsens in Norman CRISPRa.
6. Nonlinear selectors improve masked recovery in all three datasets, and MLP
   has real pancreas identity, ectopic-fill, and disease-contrast positives.
   No advanced selector dominates all biological endpoints, and the
   FT-Transformer does not improve on MLP.

## Publication gate

| Gate | Result |
|---|---|
| At least 5% lower held-out log1p MAE, CI excluding zero, in two real datasets | **Pass**: colon and pancreas |
| Calibrated selective recovery with biology preserved | **Pass, qualified**: recovery is 39–56% at exact 2% coverage and measured biology is largely preserved; corrected pancreas ectopic fill is 1.95% |
| No material degradation in deviance or correlation | **Dense output fails**; calibrated 2% not materially degraded on the biological endpoints |
| Fusion-specific benefit in multiple settings | **Fail**: +1.50 pp in pancreas, -0.28 pp in colon, and a negligible +0.06 pp in CRISPRa versus no-fusion selection |
| Preserve KO, CRISPRa, CRISPRi, and ECCITE-seq perturbation direction | **Qualified pass**: CRISPRa direction is preserved within uncertainty; Adamson CRISPRi, Dixit KO, and Papalexi ECCITE-seq global signatures improve slightly, but Adamson direct target-effect error worsens and the Papalexi target effect is neutral. |
| Improve real intervention-response GRN recovery | **Qualified pass**: edge PR AUC improves across all ten fractions for Adamson CRISPRi and Papalexi ECCITE-seq, is neutral for Dixit knockout, and worsens for Norman CRISPRa; weighted kNN remains stronger on the two positive datasets. |
| Calibrated decreasing risk under a more selective gate | **Established**: ectopic fill rises with budget (0.5% to 2.6% colon; corrected 1.95% to 5.87% pancreas) |
| Advanced selector universally improves reconstruction and biology | **Fail, with qualified positives**: MLP improves 2% masked recovery in all three datasets and several pancreas biological endpoints, but worsens canonical-marker recovery and CRISPRa signature preservation; FT-Transformer does not beat MLP. |

Decision: the evidence supports a **selective zero-filling and attribution
paper**. It does not support a broad fusion-superiority or best-imputation
claim. The positive biological contribution is the sparse decision/evaluation
framework; the fusion contribution is small and pancreas-specific.

## Source artifacts

- Colon primary bootstrap: `artifacts/paper_evidence/colon_mask_primary.json`
- Pancreas primary bootstrap: `artifacts/paper_evidence/pancreas_mask_primary.json`
- PBMC descriptive primary result: `artifacts/paper_evidence/pbmc_mask_primary.json`
- Pancreas curated biology: `artifacts/paper_evidence/pancreas_curated_biology_selective.parquet`
- PBMC independent protein analysis: `artifacts/paper_evidence/pbmc_protein_biology.parquet`
- Colon workflow run: `artifacts/colon_runs/REPLACE_WITH_RUN_ID/`
- Pancreas workflow run: `artifacts/pancreas_runs/REPLACE_WITH_RUN_ID/`
- PBMC workflow run: `artifacts/pbmc_runs/REPLACE_WITH_RUN_ID/`
- Pancreas donor cross-fit, dense: `artifacts/paper_evidence/pancreas_crossfit/results/`
- Pancreas donor cross-fit, selective curve: `artifacts/paper_evidence/pancreas_crossfit_selective/results/`
- Colon donor-level biology, dense + selective curve: `artifacts/paper_evidence/colon_donor_biology/results/`
- CRISPRa perturbation preservation (Norman): `artifacts/paper_evidence/norman_crispra/results/`
- Per-fold method contracts (including sparse guardrail): `artifacts/paper_evidence/pancreas_crossfit/fold_{0,1,2}/`
- Calibrated selector, pancreas locked-split contracts + report: `artifacts/paper_evidence/pancreas_calibrated/`
- Logistic cross-fit biology: `artifacts/paper_evidence/pancreas_crossfit_calibrated_fixed/results/`
- MLP exact-budget selection: `artifacts/paper_evidence/selector_mlp_range/`
- Calibrated selector, colon donor-level results: `artifacts/paper_evidence/colon_calibrated/results/`
- Calibrated selector, CRISPRa results: `artifacts/paper_evidence/norman_calibrated/results/`
- Matched-budget selector/value attribution: `artifacts/paper_evidence/selector_attribution/`
- Selector architecture comparison: `artifacts/paper_evidence/selector_architectures/`
- Fixed-threshold architecture deployment and biology: `artifacts/paper_evidence/selector_deployment/`
- Exact-budget logistic/MLP deployment and biology: `artifacts/paper_evidence/selector_exact_budget/`
- FT-Transformer sensitivity: `artifacts/paper_evidence/selector_architectures_ft/`
- Post-facto marker replication: `artifacts/paper_evidence/postfacto_markers/`
- Calibrated PBMC protein check: `artifacts/paper_evidence/pbmc_calibrated_protein.{parquet,json}`
- Adamson CRISPRi preparation, contracts, calibration, and evaluation: `external_data/prepared/adamson_crispri.*`; `artifacts/external_perturbseq/adamson_crispri/`
- Dixit KO preparation, contracts, calibration, and evaluation: `external_data/prepared/dixit_ko.*`; `artifacts/external_perturbseq/dixit_ko/`
- Papalexi ECCITE-seq preparation, contracts, calibration, and evaluation: `external_data/prepared/papalexi_eccite.*`; `artifacts/external_perturbseq/papalexi_eccite/`
- CellRank zebrafish trajectory preparation and evaluation: `external_data/prepared/zebrafish_trajectory.*`; `artifacts/external_trajectory/zebrafish/`
- Complete five-task real-data protocol, result tables, paired intervals, and five-panel figure: `docs/COMPLETE_DOWNSTREAM_PROTOCOL.md`; `artifacts/paper_evidence/downstream_complete/`
- Calibration script: `scripts/calibrated_selective_fill.py`
