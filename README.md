# Safe Fusion

Safe Fusion fills a chosen fraction of the recorded zeros of a single-cell RNA
sequencing count matrix and keeps every recorded count. Five teachers propose
a count for each zero: the gene median, singular value decomposition (SVD),
weighted k-nearest neighbors (kNN), MAGIC and scVI. The main mode is transductive: teachers use all input cells without cross-fitting.
A gradient-boosted regression learns the fused value from masked training
positives. A multilayer perceptron learns to rank candidate zeros from masked
training counts and recorded zeros. Safe Fusion fills the chosen top-ranked
fraction and preserves all nonzero input counts. Detection-weighted deployment
multiplies the fused value by the calibrated detection probability.

The inductive mode fits teachers on training cells and cross-fits their training
proposals before scoring held-out cells. Perturbation or donor labels can be
included where the analysis specifies them.

This repository contains the method, the benchmark, and the Slurm launchers
that reproduce every table and figure of the manuscript "Safe Fusion:
Selective Zero Imputation for Single-Cell RNA Sequencing".

## Installation

The main environment uses Python 3.11 or 3.12 and
[uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/yunweizhao26/safe-fusion.git
cd safe-fusion
uv sync --python 3.11 --extra workflow --extra paper
source .venv/bin/activate
```

The MAGIC and scVI teachers and the comparison methods use separate conda
and Python environments. [docs/REPRODUCTION.md](docs/REPRODUCTION.md)
(section 1) builds them.

## Data

The benchmarks use public data: colon epithelium and pancreatic islets from
CELLxGENE, the Norman, Adamson, Dixit and Papalexi perturbation screens, and
the zebrafish axial mesoderm time course of CellRank. Hao 2021 CITE-seq blood
data (GSE164378) support the FcRL3 check in Supplementary Section S9 and the
41 RNA-protein pairs in Section S10 / Table S34. See reproduction guide sections
24 and 36 for downloads and analysis commands.
[docs/DATA.md](docs/DATA.md) lists the sources, the prepared paths and their
checksums. The repository does not redistribute these datasets. The derived
inputs of the RNA-protein analysis (Supplementary Figure S1, Supplementary Tables S22 and
S23) are in
[data/](data/README.md), stored with Git LFS.

## Reproducing the manuscript

[docs/REPRODUCTION.md](docs/REPRODUCTION.md) gives commands for
Tables 1–3, Figures 1–3, Supplementary Figure S1 and Supplementary Tables S1–S34, in dependency order, with the output directory of each
job. [docs/PAPER_EXPERIMENTS.md](docs/PAPER_EXPERIMENTS.md) maps each table
and figure to its scripts and output files. The selector scores depend on the
processor model. The paper's selectors ran on Intel Xeon Platinum 8592+ nodes
([Conventions](docs/REPRODUCTION.md#conventions)).

| Path | Contents |
|---|---|
| `src/safefusion_benchmark/` | Data contracts, splits, masking, evaluation and the workflow command line |
| `scripts/analyses/` | Final transductive, biological-control, posterior and lupus analysis launchers |
| `scripts/` | Teachers, fused value, selector, comparison methods, evaluations and Slurm launchers |
| `workflow/`, `configs/`, `profiles/` | Snakemake workflow that prepares the colon and pancreas splits and masks |
| `data/` | Papalexi RNA-protein evaluation inputs (Git LFS) |
| `docs/` | Reproduction guide, data sources and protocols |

Generated data, models and results are written under `artifacts/`, which Git
ignores.

The manuscript compares Safe Fusion with nine published methods. SVD and
weighted kNN are Safe Fusion teachers, shown separately as an ablation.
Figure 3 and Supplementary Table S25 cover every published count-valued method
except scGPT at 19 fill fractions from 0.25% to 100%, with every method fitted
on all cells. S25 reports the gap to unthinned counts closed at 1%, 10% and
100%. Reproduction guide section 42 includes the SVD and ALRA refits.

Table 3 and Supplementary Table S23 include ten published rankings (scVI mean
and nonzero probability are separate) on the 3,400-cell benchmark and the
20,156-cell full screen. Full-screen scImpute exceeded 96 GB after 7.7 h;
scRecover and EnImpute were stopped after 11.8 and 11.6 h. These fits remain
not finished. Section 43 documents their status and the available results.
Section 44 adds scRecover and scGPT to Table 2 and Supplementary Table S17.

## Citation

See [CITATION.cff](CITATION.cff).
