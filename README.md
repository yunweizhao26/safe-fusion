# Safe Fusion

Safe Fusion fills a chosen fraction of the recorded zeros of a single-cell RNA
sequencing count matrix and keeps every recorded count. Five teachers propose
a count for each zero: the gene median, singular value decomposition (SVD),
weighted k-nearest neighbors (kNN), MAGIC and scVI. Every teacher is
inductive, and each proposal for a training cell comes from a model fitted
without that cell. A gradient-boosted regression combines the five proposals
into the value inserted into a zero. A multilayer perceptron, the selector,
learns from artificially hidden counts which zeros resemble detected entries
and ranks the zeros of new cells. Safe Fusion fills the top-ranked fraction.
When the perturbation of each cell is known, the weighted kNN teacher, the
scVI teacher and the selector can use it.

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
the zebrafish axial mesoderm time course of CellRank.
[docs/DATA.md](docs/DATA.md) lists the sources, the prepared paths and their
checksums. The repository does not redistribute these datasets. The derived
inputs of the RNA-protein analysis (Figure 2, Supplementary Tables S15 and
S16) are in
[data/](data/README.md), stored with Git LFS.

## Reproducing the manuscript

[docs/REPRODUCTION.md](docs/REPRODUCTION.md) gives the commands for every
table and figure, in dependency order, with the output directory of each
job. [docs/PAPER_EXPERIMENTS.md](docs/PAPER_EXPERIMENTS.md) maps each table
and figure to its scripts and output files. The selector scores depend on the
processor model. The paper's selectors ran on Intel Xeon Platinum 8592+ nodes
([Conventions](docs/REPRODUCTION.md#conventions)).

| Path | Contents |
|---|---|
| `src/safefusion_benchmark/` | Data contracts, splits, masking, evaluation and the workflow command line |
| `scripts/` | Teachers, fused value, selector, comparison methods, evaluations and Slurm launchers |
| `fusion/` | Autoencoder fusion network, the comparison value model of Supplementary Table S9 |
| `workflow/`, `configs/`, `profiles/` | Snakemake workflow that prepares the colon and pancreas splits and masks |
| `data/` | Papalexi RNA-protein evaluation inputs (Git LFS) |
| `docs/` | Reproduction guide, data sources and protocols |

Generated data, models and results are written under `artifacts/`, which Git
ignores.

## Citation

See [CITATION.cff](CITATION.cff).
