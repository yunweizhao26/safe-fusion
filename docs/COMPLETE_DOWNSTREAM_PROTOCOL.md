# Complete downstream validation protocol

## Scope

This protocol evaluates five downstream tasks with real biological
data. SERGIO is not part of the evidence. Every learned imputation model is fit
without its test cells. Artificially masked nonzero counts are used only to fit
and evaluate the zero selector. Biological labels, conditions, stages, and
perturbation responses are reserved for downstream evaluation.

## Common method comparison

The primary comparison includes the corrupted count matrix, gene median, SVD,
weighted nearest-neighbor regression, dense Safe Fusion, and the MLP
`apply_topk` selector at every integer selected fraction from 1 through 10 percent.
The selector receives the same three teacher outputs in every dataset. Dense
Safe Fusion is retained as a safety control rather than as the proposed method.

## Tasks and biological units

| Task | Data | Held-out unit | Primary metrics |
|---|---|---|---|
| Unsupervised clustering | Pancreatic islets and Crohn colon epithelium | Donor | Cell-type ARI, cell-type NMI, agreement with clustering from unmasked counts, neighborhood purity |
| Marker recovery | Pancreatic islets and Crohn colon epithelium | Donor | Canonical marker PR AUC, marker-effect rank correlation, off-target marker-zero filling |
| Differential expression | Pancreatic islets, T1D and autoantibody-positive versus control | Donor | Pseudobulk log-fold-change Spearman correlation, RMSE, top-effect direction preservation |
| Trajectory reconstruction | Experimental zebrafish axial mesoderm time course | Developmental stage | Stage-order Spearman and MAE, lineage balanced accuracy, adjacent-stage neighborhoods, dynamic-gene and stage-pseudobulk correlation |
| Interventional GRN | Norman CRISPRa, Adamson CRISPRi, Dixit knockout, Papalexi ECCITE-seq | Perturbed regulator | Response-edge PR AUC and ROC AUC, effect-size correlation, direction accuracy, top-edge Jaccard, source-effect error |

## Real interventional GRN definition

Each genetic intervention defines a regulator-to-response experiment. Norman
CRISPRa supplies gain-of-function interventions. Adamson CRISPRi, Dixit Cas9
knockout, and Papalexi ECCITE-seq supply loss-of-function interventions.

For every perturbed regulator, unmasked development cells are divided into two
halves. A response edge is retained when its absolute development log2 fold
change is in the top 5 or 10 percent and its direction agrees in both halves.
The perturbed source gene is excluded. Held-out perturbation and control cells
then test whether each processed matrix recovers these response edges. Thus the
benchmark uses experimentally induced transcriptional responses rather than a
simulated network. These edges represent causal perturbation responses and are
not asserted to be direct physical binding interactions.

The available processed objects do not contain independent biological replicate
fields for every perturbation screen. Perturbed regulators are therefore the
bootstrap units, and this limitation must accompany any manuscript claim.

## Failure and leakage checks

- Cell and gene order must agree across the truth, corrupted, and method outputs.
- Every learned method contract must state that test cells were not used for fitting.
- No test masking label may fit a selector or choose an operating fraction.
- Clustering uses no cell-type labels during PCA or K-means fitting.
- Marker panels are specified before examining test outputs.
- DE uncertainty resamples donors, trajectory uncertainty resamples stages, and GRN uncertainty resamples perturbed regulators.
- Results are reported across the complete 1 to 10 percent selected range rather than at one chosen point.
- The unmasked test matrix is reported as a reference ceiling, not as an available imputation method.
