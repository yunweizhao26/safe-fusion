# Downstream evaluation protocol

## Scope

This protocol defines the downstream analyses of the manuscript (main Table 4
and Supplementary Table S10). Every learned model is fitted without the
held-out cells. Artificially masked nonzero counts fit and evaluate the zero
selector only. Cell-type labels, disease status, developmental stages and
perturbation responses are reserved for evaluation.

## Compared matrices

- The masked input and, on the masked benchmark, the unmasked held-out matrix
  as a reference.
- Safe Fusion at every integer fill fraction from 1% to 10%. Five teachers
  (gene median, SVD, weighted kNN, MAGIC and scVI, all inductive and
  cross-fitted for training cells) propose counts, a gradient-boosted
  regression gives the inserted value, and the MLP selector ranks the zeros.
- SVD and weighted kNN at the same fill fractions, each ranking zeros by its
  own imputed value.
- Dense SVD, dense weighted kNN and the dense fused value, which replace every
  entry.

## Tasks and resampling units

| Task | Data | Resampling unit | Endpoints |
|---|---|---|---|
| Marker recovery | Pancreatic islets, Crohn colon epithelium | Donor | Marker AUPRC, off-target fill |
| Clustering | Pancreatic islets, Crohn colon epithelium | Donor | Adjusted Rand index, neighbor purity |
| Reference mapping | Pancreatic islets, Crohn colon epithelium | Donor | Macro F1 |
| Disease effects | Pancreatic islets, autoantibody-positive and type 1 diabetes donors against controls | Donor | Spearman correlation of pseudobulk log fold changes |
| Developmental dynamics | Zebrafish axial mesoderm time course | Stage | Dynamic-gene stage-mean Spearman correlation, stage-rank error |
| Perturbation response edges | Norman CRISPRa, Adamson CRISPRi, Dixit knockout, Papalexi ECCITE-seq | Perturbed gene | Edge AUPRC |

Supplementary Section S7 of the manuscript defines each endpoint.

## Perturbation response edges

For each perturbed gene, the held-out perturbed cells and the held-out control
cells are each divided at random into a reference half and a scoring half. In
the reference half, the log2 fold change of each gene is computed from unmasked
counts. A gene other than the perturbed gene forms a response edge when its
absolute fold change is in the top 10% and its sign agrees in two random
subsets of the reference half and in the whole reference half. Each processed
matrix ranks genes by their absolute log2 fold change in the scoring half, and
edge AUPRC is the average precision of this ranking for the reference edges. A
perturbed gene is kept when it is measured, has at least 10 perturbed
development cells and 10 perturbed held-out cells, and yields at least 10
edges. No model sees the cells that define the edges. The edges are
perturbation responses and are not claimed to be direct regulatory
interactions.

## Restored hidden counts and filled recorded zeros

On the masked benchmark, each Safe Fusion fill is split into its selected
masked positives and its selected recorded zeros, and each part is evaluated
alone (`scripts/decompose_fills.py`). For the analysis of recorded zeros, the
training cells keep the benchmark mask, the held-out cells carry their
recorded counts, every model is refitted, and each filled matrix is compared
with the unfilled recorded matrix (`scripts/build_deployment_inputs.py` and
the `scripts/slurm_deployment_*.sh` chain).

## Leakage checks

- Cell and gene order agree across the unmasked, masked and method matrices.
- Every learned method contract states that held-out cells were not used for
  fitting.
- No masking label of a held-out cell fits a selector or chooses a fill
  fraction.
- Clustering uses no cell-type labels during PCA or k-means fitting.
- Marker panels come from the source studies (`src/safefusion_benchmark/marker_panels.py`).
- Results are reported for every fill fraction from 1% to 10%.
- The unmasked held-out matrix is a reference and is not an available
  imputation method.

Commands are in [REPRODUCTION.md](REPRODUCTION.md), sections 9 (recorded
zeros, Table 4) and 13 (masked benchmark, Table S10).
