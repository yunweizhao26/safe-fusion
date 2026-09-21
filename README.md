# Safe Fusion / Signal-to-Noise

Safe Fusion combines estimates from several imputation methods, scores the
zero entries, and fills a chosen fraction with the fused estimates. Measured
nonzero entries are preserved. The paper uses an MLP to select entries to fill.

This repository contains code to reproduce:

- Masked-count recovery and F1 versus fill-rate curves.
- Marker recovery, clustering, and differential-expression analyses.
- Perturbation, trajectory, gene-regulatory-network, and RNA–protein analyses.
- Baseline comparisons and selector ablations.

Large datasets, full imputed matrices, job logs, and paper drafts are not
included. Data sources and expected paths are in [docs/DATA.md](docs/DATA.md).

## Environment

Use Python 3.11 or 3.12 and run commands from the repository root.
With `uv` installed:

```bash
git clone https://github.com/yunweizhao26/safe-fusion.git
cd safe-fusion
uv sync --python 3.11 --extra workflow --extra paper
source .venv/bin/activate
```

The workflow expects the environment at `.venv/`. PyTorch is needed for fusion
training; Matplotlib and Seaborn are needed for figures. External baselines
have separate requirements; see [workflow/envs/](workflow/envs/).

Check the installation:

```bash
snakemake --snakefile workflow/Snakefile --profile profiles/default
```

This check uses synthetic data. Paper experiments require the datasets below.

## Data setup

Download the source data listed in [docs/DATA.md](docs/DATA.md), then prepare
the inputs with the corresponding scripts:

| Dataset | Preparation script |
|---|---|
| Colon epithelium | [prepare_colon_atlas.py](scripts/prepare_colon_atlas.py) |
| Pancreatic islets | [prepare_pancreas_atlas.py](scripts/prepare_pancreas_atlas.py) |
| PBMC CITE-seq | [prepare_pbmc_citeseq.py](scripts/prepare_pbmc_citeseq.py) |
| Norman CRISPRa | [prepare_norman_crispra.py](scripts/prepare_norman_crispra.py) |
| Adamson CRISPRi, Dixit KO, Papalexi ECCITE-seq | [prepare_external_perturbseq.py](scripts/prepare_external_perturbseq.py) |
| Zebrafish trajectory | [prepare_zebrafish_trajectory.py](scripts/prepare_zebrafish_trajectory.py) |

For example, after downloading the colon source H5AD:

```bash
python scripts/prepare_colon_atlas.py \
  --input /path/to/downloaded_colon.h5ad \
  --output external_data/prepared/colon_epithelial.h5ad \
  --report external_data/prepared/colon_epithelial_preparation.json \
  --seed 1729
```

Replace `/path/to/downloaded_colon.h5ad` with your downloaded file. Each script
supports `--help`. The dataset configurations check input hashes; use the
source files and preparation settings specified in the data guide.

## Methods

- Training and inference: [run_leakage_safe_method.py](scripts/run_leakage_safe_method.py).
- MLP selection and fill-rate evaluation: [calibrated_selective_fill.py](scripts/calibrated_selective_fill.py).
- Baselines: SVD, weighted kNN, ALRA, SAVER, MAGIC, scVI, scGCL, and scGPT.
  Their entry points are listed in [docs/PAPER_EXPERIMENTS.md](docs/PAPER_EXPERIMENTS.md).

A fill rate of `0.02` means selecting 2% of candidate zero entries. The
benchmark hides 10% of measured positive counts before fitting. The selector
is fitted on development or validation cells and evaluated on held-out cells.

## Main table and masked recovery

### 1. Train the methods and evaluate reconstruction

For colon:

```bash
snakemake --snakefile workflow/Snakefile --profile profiles/default \
  --configfile configs/colon_pilot.yaml --cores 4
```

Use `configs/pancreas_pilot.yaml` or `configs/pbmc_pilot.yaml` for pancreatic
islets or PBMC. The workflow saves matrices, splits, metrics, and provenance
under `artifacts/<dataset>_runs/<run-id>/`. The run ID includes the Git commit
and configuration hash.

### 2. Select entries with Safe Fusion

Set `RUN` to the colon output directory printed by the workflow. Replace
`REPLACE_WITH_RUN_ID` below with that directory's run ID.

```bash
RUN=artifacts/colon_runs/REPLACE_WITH_RUN_ID
DATA="$RUN/data/colon_epithelial"
METHODS="$RUN/methods/standardized/colon_epithelial/mask_010"
SELECTOR=artifacts/paper_evidence/selector_mlp_range/colon

python scripts/calibrated_selective_fill.py \
  --truth "$DATA/preprocessed.h5ad" \
  --corrupted "$DATA/corrupted/mask_010.h5ad" \
  --coordinates "$DATA/coordinates/mask_010.parquet" \
  --splits "$DATA/splits.parquet" \
  --fusion-contract "$METHODS/safe_fusion" \
  --teacher-contract "$METHODS/graph_smooth" \
  --teacher-contract "$METHODS/svd_impute" \
  --teacher-contract "$METHODS/gene_median" \
  --output-dir "$SELECTOR" \
  --architecture mlp --budget-mode apply_topk \
  --fit-split validation --fit-cells 3852 \
  --budgets 0.02 0.05 0.081 0.10 \
  --curve-min-budget 0.001 --curve-max-budget 1.0 --curve-points 1000 \
  --seed 1729
```

Here, a *contract* is a directory containing a method's predicted counts and
metadata. The command saves selected matrices for each requested fill rate
and `calibration_report.json` with masked recovery and F1 curves. For example,
the 2% matrix is under `safe_fusion_calibrated_mlp_topk_0p02/`.

Reconstruction metrics are in `$RUN/tables/consolidated_metrics.parquet`.
For paired comparisons and uncertainty estimates, follow
[the locked-test analysis](docs/REPRODUCTION.md#1-primary-locked-runs-colon-pancreatic-islets-pbmc).
Pancreas uses three donor folds, and Norman uses a perturbation split; follow
their preparation and fitting steps in [docs/REPRODUCTION.md](docs/REPRODUCTION.md).

## F1 versus fill-rate curves

The paper compares pancreatic islets, colon, and Norman CRISPRa across
1,000 fill rates from 0.1% to 100%. Prepare the three datasets, train every
baseline, and run MLP selection for each pancreas fold, colon, and Norman.
The dataset settings are in [slurm_selector_mlp_range.sh](scripts/slurm_selector_mlp_range.sh).

The curve scripts expect the artifact layout recorded in
[docs/REPRODUCTION.md](docs/REPRODUCTION.md). Set `PANCREAS` and `COLON` in
[compute_matched_baseline_f1_curves.py](scripts/compute_matched_baseline_f1_curves.py)
to your workflow run directories before running these commands. All baseline
matrices and selector reports must be present.

```bash
python scripts/compute_matched_baseline_f1_curves.py
python scripts/combine_mlp_baseline_curves.py
python scripts/plot_selector_f1_fillrate.py \
  --input artifacts/paper_evidence/selector_f1_fillrate_mlp_baselines_1000_points.csv \
  --output artifacts/paper_evidence/f1_fillrate.png
```

The combined CSV contains the per-method curves. The companion
`selector_f1_fillrate_mlp_baselines_summary.json` contains range summaries.

## Biological experiments and ablations

Run each experiment after preparing its data and generating the method
matrices. The linked instructions give the required inputs and evaluation
commands; the [experiment map](docs/PAPER_EXPERIMENTS.md) lists the scripts.

| Experiment | Instructions |
|---|---|
| Colon markers and donor-level biology | [Colon evaluation](docs/REPRODUCTION.md#2-colon-donor-level-biology-uses-existing-contracts-no-refit) |
| Pancreas markers and donor cross-validation | [Pancreas folds](docs/REPRODUCTION.md#3-pancreatic-islet-donor-cross-fit-dense--selective) |
| Marker recovery, clustering, and differential expression across fill rates | [Downstream benchmark](docs/REPRODUCTION.md#complete-five-task-real-data-downstream-benchmark) |
| Norman target-expression recovery | [CRISPRa benchmark](docs/REPRODUCTION.md#4-norman-crispra-perturbation-benchmark) |
| Adamson, Dixit, and Papalexi perturbation responses | [External screens](docs/REPRODUCTION.md#external-crispri-ko-and-eccite-seq-screens) |
| Zebrafish trajectory preservation | [Trajectory benchmark](docs/REPRODUCTION.md#zebrafish-trajectory) |
| Simulated and perturbation-based gene-regulatory-network analysis | [GRN entry points](docs/PAPER_EXPERIMENTS.md) |
| PBMC RNA–protein agreement | [Protein evaluation](docs/REPRODUCTION.md#post-facto-markers-and-pbmc-protein) |
| Papalexi RNA–protein agreement | [Cross-modal evaluation](docs/REPRODUCTION.md#external-crispri-ko-and-eccite-seq-screens) |
| Selector architecture and component ablations | [Matched-budget attribution](docs/REPRODUCTION.md#matched-budget-selectorvalue-attribution) |

Use `--architecture mlp --budget-mode apply_topk` for the paper's selector.
Use the other architectures only for the corresponding ablations. The Slurm
launchers contain resource requests and fixed artifact paths; adapt these to
your environment. Submit from the repository root and supply your cluster's
account and partition with `sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION`.
Their Python commands can also be run directly from the repository root.

## Results

Workflow outputs are saved under `artifacts/<dataset>_runs/<run-id>/`.
Paper summaries and figures are saved under `artifacts/paper_evidence/`.
Keep the configuration, seed, run manifest, and `provenance/` files with each
result so that its inputs and commands can be traced.

Generated data and results are Git-ignored. Local material outside the paper
is kept in `archive/`, which is also Git-ignored.
