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
the zebrafish axial mesoderm time course of CellRank.
[docs/DATA.md](docs/DATA.md) lists the sources, the prepared paths and their
checksums. The repository does not redistribute these datasets. The derived
inputs of the RNA-protein analysis (Figure 2, Supplementary Tables S19 and
S20) are in
[data/](data/README.md), stored with Git LFS.

## Reproducing the manuscript

[docs/REPRODUCTION.md](docs/REPRODUCTION.md) gives the commands for every
Table 1–3, Figure 1–2 and Supplementary Table S1–S29, in dependency order, with the output directory of each
job. [docs/PAPER_EXPERIMENTS.md](docs/PAPER_EXPERIMENTS.md) maps each table
and figure to its scripts and output files. The selector scores depend on the
processor model. The paper's selectors ran on Intel Xeon Platinum 8592+ nodes
([Conventions](docs/REPRODUCTION.md#conventions)).

| Path | Contents |
|---|---|
| `src/safefusion_benchmark/` | Data contracts, splits, masking, evaluation and the workflow command line |
| `scripts/analyses/` | Final transductive, biological-control, posterior and lupus analysis launchers |
| `scripts/` | Teachers, fused value, selector, comparison methods, evaluations and Slurm launchers |
| `fusion/` | Autoencoder fusion network, the comparison value model of Supplementary Table S12 |
| `workflow/`, `configs/`, `profiles/` | Snakemake workflow that prepares the colon and pancreas splits and masks |
| `data/` | Papalexi RNA-protein evaluation inputs (Git LFS) |
| `docs/` | Reproduction guide, data sources and protocols |

Generated data, models and results are written under `artifacts/`, which Git
ignores.

## Citation

See [CITATION.cff](CITATION.cff).
