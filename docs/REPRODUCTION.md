# Reproducing the Safe Fusion manuscript

This file gives the commands that reproduce every table and figure of the
manuscript "Safe Fusion: Selective Zero Imputation for Single-Cell RNA
Sequencing" (Briefings in Bioinformatics). Sections 1 to 3 build the
environments, prepare the data, and fit Safe Fusion and the comparison
methods. Sections 4 to 17 each produce one manuscript item, or two items
that come from the same jobs. Every job depends only on jobs submitted
earlier in this file, so the sections follow the job dependencies and not
the order of the manuscript.
[PAPER_EXPERIMENTS.md](PAPER_EXPERIMENTS.md) lists the scripts and output
files of every item.

| Manuscript item | Section |
|---|---|
| Table 1, Figure 1 | [4](#4-table-1-and-figure-1-masked-recovery-at-matched-fill-fractions) |
| Table 2, Supplementary Table S7 | [8](#8-table-2-and-supplementary-table-s7-accuracy-of-the-inserted-value) |
| Table 3, Supplementary Table S8 | [11](#11-table-3-and-supplementary-table-s8-zeros-created-by-knockdown) |
| Table 4 | [9](#9-table-4-filling-recorded-zeros-in-held-out-cells) |
| Figure 2, Supplementary Table S9 | [12](#12-figure-2-and-supplementary-table-s9-agreement-with-surface-protein) |
| Supplementary Table S1 | [14](#14-supplementary-table-s1-fill-decisions-of-standard-imputers) |
| Supplementary Table S2 | [15](#15-supplementary-table-s2-benchmark-data) |
| Supplementary Table S3 and Section S3 | [16](#16-supplementary-table-s3-and-section-s3-comparison-methods) |
| Supplementary Table S4 | [5](#5-supplementary-table-s4-mask-and-seed-replicates) |
| Supplementary Table S5 | [6](#6-supplementary-table-s5-selector-feature-groups) |
| Supplementary Table S6 | [7](#7-supplementary-table-s6-binomial-thinning) |
| Supplementary Table S10 | [13](#13-supplementary-table-s10-downstream-analyses-on-the-masked-benchmark) |
| Supplementary Table S11 | [10](#10-supplementary-table-s11-fill-fraction-rule) |
| Supplementary Table S12 | [17](#17-supplementary-table-s12-runtime) |

## Conventions

- Run every command from the repository root. Each launcher changes to
  `SLURM_SUBMIT_DIR` and writes its log to `logs/`, so create that
  directory first (`mkdir -p logs`).
- All compute runs through Slurm. Some launchers carry
  `#SBATCH --account=torch_pr_634_general`, the account of the cluster where
  the paper was run. Set your own account (and partition, if your cluster
  needs one) once per shell. Slurm input environment variables override
  `#SBATCH` lines, and they also apply to the `sbatch` calls inside
  `scripts/slurm_thinning_benchmark.sh`:

  ```bash
  export SBATCH_ACCOUNT=<account>
  export SBATCH_PARTITION=<partition>   # only if your cluster needs it
  ```

- Python commands that have no launcher also run on a compute node, for
  example inside `srun`. They use the environment of section 1
  (`source .venv/bin/activate`).
- Every scVI fit requests an NVIDIA L40S GPU (`--gres=gpu:l40s:1`), because
  scVI outputs differ between GPU types. This applies to
  `slurm_scvi_teachers.sh`, `slurm_deployment_scvi.sh`,
  `slurm_scvi_current_baselines.sh`, `slurm_papalexi_crossmodal_scvi.sh`, and
  to the scVI stages of `slurm_seed_replicates.sh`,
  `slurm_thinning_benchmark.sh` and `slurm_condition_aware_screens.sh`, whose
  `sbatch` commands below request the same GPU. On other GPU types the
  scVI values, and the results that use them, change slightly.
- Every ALRA launcher (`slurm_alra_colon.sh`, `slurm_alra_pancreas_crossfit.sh`,
  `sbatch_run_alra_norman.s`) fixes `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`
  and `MKL_NUM_THREADS` at 8, because the randomized SVD of ALRA depends on
  the number of BLAS threads.
- The launchers and scripts fix their seeds. Most splits, masks, models and
  bootstraps use seed 1729. The mask replicates of section 5 use seeds 1730
  to 1733, and the thinning bootstrap of section 7 uses seed 7. scGCL keeps
  the upstream seed 0. The package versions are pinned in
  `uv.lock`, `workflow/envs/` and `scripts/standard_imputers/environment.yaml`.
  Numerical results can still vary across hardware and library builds.
- Each section stores job IDs in shell variables (`teachers`, `scvi`,
  `stack`, ...), and later sections depend on them. Submit the sections in
  order from one shell, or drop a dependency on a job that has already
  finished.
- The launchers read the colon and pancreas workflow runs of section 2.2 from
  the fixed directories `artifacts/colon_runs/0b2469810675-c0db6f963e94` and
  `artifacts/pancreas_runs/0b2469810675-45c81b160d78`.
- `artifacts/`, `external_data/`, `logs/` and the environments are ignored by
  Git.

## 1. Environments

```bash
uv sync --python 3.11 --extra workflow --extra paper
source .venv/bin/activate
mkdir -p logs

sbatch scripts/slurm_setup_current_baselines.sh    # .conda-scvi-current and .conda-magic-current
sbatch scripts/sbatch_setup_r_baselines.s          # .conda-r-baselines
sbatch scripts/sbatch_setup_python_baselines.s     # .venv-baselines and the scGPT checkpoint

git clone https://github.com/zehaoxiong123/scGCL.git baselines_and_data/scGCL
git -C baselines_and_data/scGCL checkout 317015acdf06d2929c20a7d2858bac539b3d8ebd
```

Python 3.12 also works for `.venv`. The environments run the following
steps.

| Environment | Built from | Runs |
|---|---|---|
| `.venv` | `pyproject.toml`, `uv.lock` | Workflow, gene median, SVD and weighted kNN teachers, fused value, selector, ALRA (Python port), SAVER driver, autoencoder fusion network, evaluations and Figure 1 |
| `.conda-scvi-current` | `workflow/envs/scvi.yaml` | scVI teacher, standard scVI, Papalexi RNA-protein preparation and audit |
| `.conda-magic-current` | `workflow/envs/magic.yaml` | MAGIC teacher, standard MAGIC |
| `.conda-r-baselines` | `scripts/sbatch_setup_r_baselines.s` | SAVER (`scripts/run_saver_baseline.R`, called by `run_saver_baseline.py --rscript`) |
| `.venv-baselines` | `scripts/sbatch_setup_python_baselines.s` | scGPT (checkpoint in `external_data/baselines/scgpt_human`), the scGCL adapter (which also imports `scanpy` and `faiss`), and the Figure 2 plot |
| `.conda-standard-imputers` | `scripts/standard_imputers/environment.yaml` | The six imputers of Supplementary Table S1, created by `slurm_standard_imputers.sh` on first use |

## 2. Data preparation

[DATA.md](DATA.md) lists the public sources and the SHA-256 of every input.
Run the commands of sections 2.1 to 2.5 on a compute node.

### 2.1 Colon epithelium and pancreatic islets

Download the two CELLxGENE files to `external_data/cellxgene/`, from
`https://datasets.cellxgene.cziscience.com/<artifact>.h5ad` with artifacts
`63ff2c52-cb63-44f0-bac3-d0b33373e312` (colon) and
`f89a618b-fe4b-404e-bd39-7c574529b1f5` (pancreas). The paper used source files
with SHA-256 `612e03925bd5860e1ec81b0295af40a7f9e4111638f8f41ccb2fe976f428f053`
(colon) and `7f0302126d35301770ba8a21eb773f80fe576b03617b441f92f7f6d1f114f0a5`
(pancreas).

```bash
python scripts/prepare_colon_atlas.py \
  --input external_data/cellxgene/63ff2c52-cb63-44f0-bac3-d0b33373e312.h5ad \
  --output external_data/prepared/colon_epithelial.h5ad \
  --report external_data/prepared/colon_epithelial.report.json --seed 1729
python scripts/prepare_pancreas_atlas.py \
  --input external_data/cellxgene/f89a618b-fe4b-404e-bd39-7c574529b1f5.h5ad \
  --output external_data/prepared/pancreas_islets.h5ad \
  --report external_data/prepared/pancreas_islets.report.json --seed 1729
```

Both scripts keep at most 12 cells per donor, cell type and condition
(`--cells-per-stratum`) and select 1,200 variable genes
(`--variable-genes`), plus a fixed marker list.

### 2.2 Workflow runs: splits and corruptions of colon and pancreas

```bash
snakemake --snakefile workflow/Snakefile --profile profiles/default \
  --configfile configs/colon_pilot.yaml --cores 4
snakemake --snakefile workflow/Snakefile --profile profiles/default \
  --configfile configs/pancreas_pilot.yaml --cores 4
```

The workflow checks the SHA-256 of each prepared file against its config and
writes, under `artifacts/{colon,pancreas}_runs/<run-id>/data/<dataset>/`,
`preprocessed.h5ad`, `splits.parquet`, `corrupted/<corruption>.h5ad` and
`coordinates/<corruption>.parquet`. The corruptions are `mask_010` (both
datasets) and the colon `thinning_050` and `thinning_025` of section 7. The
manuscript uses only these data outputs. The DAG also fits and evaluates the
simple methods enabled in the configs, whose results the manuscript does not
report, and the dense `safe_fusion` entry is disabled. Section 3 writes the
colon teachers into the workflow directory
`methods/standardized/colon_epithelial/mask_010/` and replaces the
workflow's `gene_median`, `svd_impute` and `graph_smooth` contracts there.

The run ID combines the Git commit and the configuration hash, so a new run
gets a new ID. Link the new run directories to the names that the launchers
read:

```bash
ln -sfn <colon run id> artifacts/colon_runs/0b2469810675-c0db6f963e94
ln -sfn <pancreas run id> artifacts/pancreas_runs/0b2469810675-45c81b160d78
```

### 2.3 Pancreatic-islet donor folds

Three donor folds, stratified by disease status, hold out each pancreas donor
once:

```bash
python scripts/make_pancreas_crossfit_splits.py \
  --input external_data/prepared/pancreas_islets.h5ad \
  --output-dir artifacts/paper_evidence/pancreas_crossfit --folds 3 --seed 1729
```

The script writes `fold_<k>/splits.parquet` and `manifest.json`.

### 2.4 Norman CRISPR activation screen

The input is the GEARS-processed Norman file (`perturb_processed.h5ad`) with
its raw UMI counts layer.

```bash
python scripts/prepare_norman_crispra.py \
  --input <path to perturb_processed.h5ad> \
  --output external_data/prepared/norman_crispra.h5ad \
  --seed 1729 --max-cells-per-condition 60 --max-control-cells 1000 \
  --min-cells-per-condition 60 --target-log2fc-min 0.5 --target-p-max 0.05
python scripts/setup_norman_crispra_experiment.py \
  --input external_data/prepared/norman_crispra.h5ad \
  --output-dir artifacts/paper_evidence/norman_crispra \
  --test-fraction 0.30 --mask-fraction 0.10 --seed 1729
```

`prepare_norman_crispra.py` reads the source with bounded h5py block reads.
Opening the source with AnnData backed fancy indexing loads the full dense
matrix.

### 2.5 Adamson, Dixit and Papalexi screens and the zebrafish time course

The screen inputs are the harmonized raw-count H5AD files of scPerturb
(Zenodo record 10044268) in `external_data/scperturb/`. The zebrafish axial
mesoderm file of CellRank goes to `external_data/trajectory/`.

```bash
for spec in \
  "adamson_crispri AdamsonWeissman2016_GSM2406681_10X010.h5ad" \
  "dixit_ko DixitRegev2016.h5ad" \
  "papalexi_eccite PapalexiSatija2021_eccite_RNA.h5ad"; do
  set -- $spec
  python scripts/prepare_external_perturbseq.py \
    --input external_data/scperturb/$2 \
    --output external_data/prepared/$1.h5ad --dataset $1
  python scripts/setup_norman_crispra_experiment.py \
    --input external_data/prepared/$1.h5ad \
    --output-dir artifacts/external_perturbseq/$1 \
    --split-column preassigned_split
done

python scripts/prepare_zebrafish_trajectory.py \
  --input external_data/trajectory/zebrafish_embryogenesis_axial_mesoderm.h5ad \
  --output external_data/prepared/zebrafish_trajectory.h5ad
python scripts/setup_norman_crispra_experiment.py \
  --input external_data/prepared/zebrafish_trajectory.h5ad \
  --output-dir artifacts/external_trajectory/zebrafish \
  --split-column preassigned_split
```

Each preparation report (`external_data/prepared/<dataset>.report.json`)
records the source SHA-256.

### 2.6 Papalexi RNA-protein benchmark

```bash
pdl1_prep=$(sbatch --parsable scripts/slurm_prepare_papalexi_crossmodal.sh)
fetch=$(sbatch --parsable scripts/slurm_fetch_papalexi_multimodal.sh)
audit=$(sbatch --parsable --dependency=afterok:${fetch} scripts/slurm_audit_papalexi_multimodal.sh)
```

- `slurm_prepare_papalexi_crossmodal.sh` writes
  `external_data/prepared/papalexi_eccite_crossmodal.h5ad`, which keeps CD274,
  CD86, PDCD1LG2 and HAVCR2, and the masked benchmark
  `artifacts/paper_evidence/papalexi_crossmodal/benchmark/` (`corrupted.h5ad`,
  `coordinates.parquet`, `splits.parquet`).
- `slurm_fetch_papalexi_multimodal.sh` downloads the public MuData object and
  the GEO antibody counts to `external_data/papalexi_multimodal/`, checks the
  MD5 of the MuData object and records the SHA-256 of the GEO archive.
- `slurm_audit_papalexi_multimodal.sh` matches the antibody counts to the RNA
  cells and writes
  `artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet`.

## 3. Safe Fusion and the comparison methods

### 3.1 Dataset units

`scripts/unit_paths.sh` defines the ten dataset units of the launchers, in
array order: `pancreas_0`, `pancreas_1`, `pancreas_2` (the donor folds),
`colon`, `norman_crispra`, `adamson_crispri`, `dixit_ko`, `papalexi_eccite`,
`zebrafish` and `papalexi_crossmodal`. `unit_paths <key>` sets `input`,
`coordinates`, `splits`, `truth` and `methods_root`, and
`teacher_contract_args <root>` expands the five teacher contracts under
`<root>`. A contract is a directory with `mean.npy` (cells by genes) and
`metadata.json`.

### 3.2 Teachers, fused value and selector

Five teachers propose a count for every entry: the gene median, SVD, weighted
kNN, MAGIC and scVI. `scripts/run_leakage_safe_method.py --method
gene_median|svd_impute|graph_smooth` fits the first three, and
`scripts/run_inductive_teacher.py --method magic|scvi` fits the inductive
MAGIC and scVI teachers. SVD, MAGIC and scVI are cross-fitted over five
partitions of the training cells (`training_folds` in
`src/safefusion_benchmark/splits.py`), and weighted kNN leaves each training
cell out of its own neighbor set, so no proposal for a training cell uses that
cell's counts.

`run_leakage_safe_method.py --method safe_fusion` fits the fused value, a
gradient-boosted regression of the log count on the masked positives of the
training cells (`--value-model boosted`, the default). It takes the five
teacher contracts with `--teacher-contract`. `--value-model linear` fits the
linear combination of Table 2 instead. The contract metadata records the
five-fold cross-validated error of both value models
(`parameters.value_model_cross_validation`).

`scripts/calibrated_selective_fill.py --architecture mlp --budget-mode
apply_topk` fits the MLP selector on the selector-fitting cells (the
validation donors in colon, the development cells elsewhere) and fills the
top-ranked zeros of the held-out cells with the fused value at each fill
fraction. It writes one `safe_fusion_calibrated_mlp_topk_<b>` contract per
fraction, where `<b>` is the fraction with `p` in place of the decimal point
(`0p01` to `0p1`), and `calibration_report.json` with the 1000-point masked
F1 curve from 0.1% to 100%. With `--condition-column target` (section 11),
the selector adds the two features computed within each perturbation label.
Held-out labels do not fit any model or choose any entry.

```bash
teachers=$(sbatch --parsable --dependency=afterok:${pdl1_prep} scripts/slurm_prepare_complete_downstream_teachers.sh)
scvi=$(sbatch --parsable --dependency=afterok:${pdl1_prep} scripts/slurm_scvi_teachers.sh)
stack=$(sbatch --parsable --dependency=afterok:${teachers}:${scvi} scripts/slurm_safe_fusion_stack.sh)
selectors=$(sbatch --parsable --dependency=afterok:${stack} scripts/slurm_complete_downstream_selectors.sh)
matched=$(sbatch --parsable --dependency=afterok:${teachers} scripts/slurm_apply_fill_fraction.sh)
pdl1_selector=$(sbatch --parsable --dependency=afterok:${stack} scripts/slurm_papalexi_crossmodal_mlp.sh)
pdl1_scvi=$(sbatch --parsable --dependency=afterok:${pdl1_prep} scripts/slurm_papalexi_crossmodal_scvi.sh)
```

- `slurm_prepare_complete_downstream_teachers.sh` (array 0-39, task =
  4 × unit + teacher) fits the gene median, SVD, weighted kNN and MAGIC
  teachers of every unit. `slurm_scvi_teachers.sh` (array 0-9) fits the scVI
  teacher. The contracts are
  `<methods_root>/{gene_median,svd_impute,graph_smooth,magic_inductive,scvi_inductive}`.
- `slurm_safe_fusion_stack.sh` (array 0-9) writes the fused value to
  `<methods_root>/safe_fusion`. With `VALUE_MODEL=linear` it writes
  `<methods_root>/safe_fusion_linear` (section 8).
- `slurm_complete_downstream_selectors.sh` (array 0-8, every unit except
  `papalexi_crossmodal`, in the order `pancreas_0`, `pancreas_1`,
  `pancreas_2`, `norman_crispra`, `adamson_crispri`, `dixit_ko`,
  `papalexi_eccite`, `zebrafish`, `colon`) fits the selector and writes the
  fills at 1% to 10%. The output directories are
  `artifacts/paper_evidence/selector_mlp_biology_range/colon`,
  `artifacts/paper_evidence/pancreas_crossfit/fold_<k>/selector_mlp_biology_range_fullteachers`,
  `artifacts/paper_evidence/selector_mlp_biology_range_fullteachers/norman_crispra`
  and `<methods_root>/selector_mlp_biology_range` for the three other screens
  and zebrafish.
- `slurm_apply_fill_fraction.sh` (array 0-8) fills SVD and weighted kNN at
  the same fractions, each ranking zeros by its own value
  (`scripts/apply_fill_fraction.py`), and writes `svd_topk_<b>` and
  `weighted_knn_topk_<b>` under
  `artifacts/paper_evidence/matched_fraction/colon`,
  `artifacts/paper_evidence/pancreas_crossfit/fold_<k>/matched_fraction`,
  `artifacts/paper_evidence/norman_crispra/matched_fraction` and
  `<methods_root>/matched_fraction` for the other screens and zebrafish.
- `slurm_papalexi_crossmodal_mlp.sh` fits the selector of the RNA-protein
  benchmark and writes
  `artifacts/paper_evidence/papalexi_crossmodal/benchmark/mlp_selector/`,
  including `selected_gene_scores.parquet` with the scores of the zeros of
  CD274, CD86, PDCD1LG2 and HAVCR2.
- `slurm_papalexi_crossmodal_scvi.sh` fits standard scVI on the RNA-protein
  benchmark (`benchmark/scvi`).

### 3.3 Comparison methods

```bash
alra_colon=$(sbatch --parsable scripts/slurm_alra_colon.sh)
alra_pancreas=$(sbatch --parsable scripts/slurm_alra_pancreas_crossfit.sh)
alra_norman=$(sbatch --parsable scripts/sbatch_run_alra_norman.s)
saver_tissue=$(sbatch --parsable scripts/sbatch_run_saver_pc.s)
saver_norman=$(sbatch --parsable scripts/slurm_saver_full_norman.sh)
saver_rescale=$(sbatch --parsable --dependency=afterok:${saver_tissue}:${saver_norman} scripts/slurm_rescale_saver_contracts.sh)
magic_std=$(sbatch --parsable scripts/slurm_magic_current_baselines.sh)
scvi_std=$(sbatch --parsable scripts/slurm_scvi_current_baselines.sh)
scgpt=$(sbatch --parsable scripts/slurm_scgpt_gene_current_baselines.sh)
scgcl=$(sbatch --parsable scripts/sbatch_scgcl_baselines.s)
scgcl_norman=$(sbatch --parsable scripts/slurm_scgcl_norman.sh)
baselines=${alra_colon}:${alra_pancreas}:${alra_norman}:${saver_rescale}:${magic_std}:${scvi_std}:${scgpt}:${scgcl}:${scgcl_norman}
```

| Launcher | Method | Output |
|---|---|---|
| `slurm_alra_colon.sh` | ALRA, colon | `artifacts/paper_evidence/baselines/alra/colon_mask_010` |
| `slurm_alra_pancreas_crossfit.sh` (array 0-2) | ALRA, pancreas folds | `artifacts/paper_evidence/pancreas_crossfit/fold_<k>/alra` |
| `sbatch_run_alra_norman.s` | ALRA, Norman | `artifacts/paper_evidence/baselines/alra/norman_mask_010` |
| `sbatch_run_saver_pc.s` | SAVER, pancreas and colon | `artifacts/paper_evidence/baselines/saver/{pancreas,colon}_mask_010` |
| `slurm_saver_full_norman.sh` | SAVER, Norman | `artifacts/paper_evidence/baselines/saver/norman_full_mask_010` |
| `slurm_magic_current_baselines.sh` (array 0-2) | Standard MAGIC | `artifacts/paper_evidence/baselines/magic/{pancreas,colon,norman}` |
| `slurm_scvi_current_baselines.sh` (array 0-2) | Standard scVI | `artifacts/paper_evidence/baselines/scvi/{pancreas,colon,norman}` |
| `slurm_scgpt_gene_current_baselines.sh` (array 0-2) | scGPT masked value decoder | `artifacts/paper_evidence/baselines/scgpt_mvc/{pancreas,colon,norman}` |
| `sbatch_scgcl_baselines.s` (array 0-1) | scGCL, pancreas and colon | `artifacts/paper_evidence/baselines/scgcl/{pancreas,colon}` |
| `slurm_scgcl_norman.sh` | scGCL, Norman | `artifacts/paper_evidence/baselines/scgcl/norman` |

ALRA is fitted on the training cells and applied to the held-out cells
(`scripts/run_alra_baseline.py`, a Python port of ALRA). SAVER, MAGIC, scVI
and scGCL are fitted on the whole masked matrix. `run_saver_baseline.py`
writes SAVER on the count scale and marks the contract
`count_scale_rescaled`. `slurm_rescale_saver_contracts.sh` puts SAVER
contracts that lack this mark on the count scale and skips the others.
`scripts/masked_f1_units.py` defines the evaluation units, the comparator
contract paths and the conversion of each contract to the count scale.

## 4. Table 1 and Figure 1: masked recovery at matched fill fractions

```bash
stacked=$(sbatch --parsable --dependency=afterok:${teachers}:${scvi}:${baselines} scripts/slurm_stacked_selector_baselines.sh)
sbatch --dependency=afterok:${selectors}:${stacked}:${baselines} scripts/slurm_matched_baseline_figure.sh
```

`slurm_stacked_selector_baselines.sh` (array 0-4: `pancreas_0`, `pancreas_1`,
`pancreas_2`, `colon`, `norman_crispra`) trains the Safe Fusion selector on one
comparator's value plus the three context features, for every comparator with
values on the selector-fitting cells. It writes
`artifacts/paper_evidence/stacked_selector_baselines/<unit>/<comparator>/`
(`report.json`, `unit_counts.parquet`). `slurm_matched_baseline_figure.sh`
runs `compute_matched_baseline_f1_curves.py`, `combine_mlp_baseline_curves.py`,
`paired_masked_f1_bootstrap.py` and `plot_selector_f1_fillrate.py`. It writes,
under `artifacts/paper_evidence/`:

- `selector_f1_fillrate_baselines_1000_points.csv`,
  `selector_f1_fillrate_baselines_summary.json` and
  `masked_f1_unit_counts.parquet`: the comparator curves and the per-unit
  counts at 1% to 10%.
- `selector_f1_fillrate_mlp_baselines_1000_points.csv` and
  `selector_f1_fillrate_mlp_baselines_summary.json`: the same curves with
  Safe Fusion and the stacked selectors. In the summary,
  `<dataset>/main/0.001_0.100/fraction_mlp_best` is the share of the 100 fill
  fractions from 0.1% to 10% at which Safe Fusion has the highest F1 (last row
  of Table 1).
- `masked_f1_paired_bootstrap.csv` and `masked_f1_paired_bootstrap.json`:
  Safe Fusion minus each comparator and each stacked selector, at 1%, 2%, 5%
  and 10% and averaged over the fill fractions 1% to 10%, with intervals from
  2000 draws of donors or perturbation targets (Table 1).
- Figure 1: `figures/f1_fillrate_3panel_oup.png`. `FIGURE=<path>` sets
  another output path.

## 5. Supplementary Table S4: mask and seed replicates

Each replicate draws a new stratified 10% mask (`make_mask_replicate.py`) and
refits the five teachers, the fused value, the selector, and standard MAGIC
and scVI with seeds 1730 to 1733. Seed 1729 is the run of sections 3 and 4.
The pancreas replicates reuse the three donor folds. `STAGE` selects the step.

```bash
SR=scripts/slurm_seed_replicates.sh
mask_rep=$(sbatch --parsable --array=0-11 --export=ALL,STAGE=mask $SR)
teachers_rep=$(sbatch --parsable --array=0-79 --dependency=afterok:${mask_rep} --export=ALL,STAGE=teachers $SR)
scvi_teacher_rep=$(sbatch --parsable --array=0-19 --gres=gpu:l40s:1 --dependency=afterok:${mask_rep} --export=ALL,STAGE=scvi_teacher $SR)
stack_rep=$(sbatch --parsable --array=0-19 --dependency=afterok:${teachers_rep}:${scvi_teacher_rep} --export=ALL,STAGE=stack $SR)
rep_lin=$(sbatch --parsable --array=0-19 --dependency=afterok:${stack_rep} --export=ALL,STAGE=stack,VALUE_MODEL=linear $SR)
selector_rep=$(sbatch --parsable --array=0-19 --dependency=afterok:${stack_rep} --export=ALL,STAGE=selector $SR)
magic_rep=$(sbatch --parsable --array=0-11 --dependency=afterok:${mask_rep} --export=ALL,STAGE=magic $SR)
scvi_rep=$(sbatch --parsable --array=0-11 --gres=gpu:l40s:1 --dependency=afterok:${mask_rep} --export=ALL,STAGE=scvi $SR)
```

`rep_lin` writes the linear combination of each replicate
(`safe_fusion_linear`) for section 8. After these jobs and the jobs
`selectors`, `magic_std` and `scvi_std` finish, run on a compute node:

```bash
python scripts/summarize_seed_replicates.py
```

The replicate inputs and models are in
`artifacts/paper_evidence/seed_replicates/seed_<seed>/`. Table S4 comes from
`artifacts/paper_evidence/seed_replicates/summary/`
(`seed_replicate_summary.csv`, `seed_replicate_across.csv`,
`seed_replicate_f1_curves.csv`).

## 6. Supplementary Table S5: selector feature groups

```bash
ablation=$(sbatch --parsable --dependency=afterok:${stack} scripts/slurm_selector_mlp_attribution_range.sh)
sbatch --dependency=afterok:${ablation} scripts/slurm_pair_mlp_attribution_range.sh
```

`slurm_selector_mlp_attribution_range.sh` (array 0-4: the pancreas folds,
Norman, colon) refits the MLP selector with all features, the teacher
features only and the context features only (`VARIANTS`, default
`full teacher_only context_only`) and fills 1% to 10% of the zeros. For the
full selector it also inserts each teacher's value into the same selected
zeros. `slurm_pair_mlp_attribution_range.sh` (array 0-2) pools the pancreas
folds and compares every variant with the full selector. The outputs are in
`artifacts/paper_evidence/selector_mlp_attribution_range/{pancreas/fold_<k>,colon,norman_crispra}/`,
and `ranking_metrics.parquet` holds the test precision-recall AUC and the
masked positive counts of each variant (Table S5). `paired/{pancreas,colon,norman_crispra}/`
holds the pooled comparisons.

## 7. Supplementary Table S6: binomial thinning

The positives are held-out entries that are nonzero in the recorded counts
and zero after thinning. Colon uses the workflow corruptions `thinning_050`
and `thinning_025` (section 2.2), and `prepare_thinning_benchmark.py` thins
pancreas to 50% of the molecules. Every model is refitted on the thinned
training cells, and standard MAGIC and scVI are fitted on all cells as
comparators. The `submit` stage calls `sbatch` for the stages `prepare`,
`cpu`, `gpu` (on an L40S), `selector` and `evaluate` with their dependencies
and prints their job IDs. The `stack` stage with `VALUE_MODEL=linear` writes
the linear combination of section 8.

```bash
read -r thin_prep thin_cpu thin_gpu thin_sel thin_eval <<< \
  "$(bash scripts/slurm_thinning_benchmark.sh submit | sed 's/[a-z]*=//g')"
thin_lin=$(sbatch --parsable --job-name=sf-thin-stack --array=0-4 \
  --dependency=afterok:${thin_cpu}:${thin_gpu} --export=ALL,VALUE_MODEL=linear \
  scripts/slurm_thinning_benchmark.sh stack)
```

`bash scripts/slurm_thinning_benchmark.sh submit` prints
`prepare=<id> cpu=<id> gpu=<id> selector=<id> evaluate=<id>`, and the `read`
line stores these IDs. The outputs are under
`artifacts/paper_evidence/thinning/`: the inputs in `data/<dataset>/`, the
teachers, fused value and selector of each unit in `<unit>/`, the comparators
in `comparators/<dataset>/`, and Table S6 in
`evaluation/matched_fraction_summary.{csv,json}`.

## 8. Table 2 and Supplementary Table S7: accuracy of the inserted value

```bash
stack_lin=$(sbatch --parsable --array=0-4 --dependency=afterok:${teachers}:${scvi} \
  --export=ALL,VALUE_MODEL=linear scripts/slurm_safe_fusion_stack.sh)
autoenc=$(sbatch --parsable --dependency=afterok:${teachers}:${scvi} scripts/slurm_autoencoder_fusion.sh)
sbatch --dependency=afterok:${stack}:${stack_lin}:${autoenc}:${selectors}:${selector_rep}:${rep_lin}:${thin_sel}:${thin_lin} \
  scripts/slurm_value_accuracy.sh
```

- `stack_lin` writes the linear combination of the teachers
  (`<methods_root>/safe_fusion_linear`) for units 0-4 (the pancreas folds,
  colon and Norman).
- `slurm_autoencoder_fusion.sh` (array 0-9, task = 2 × unit + mode) fits the
  autoencoder fusion network of `fusion/` through
  `scripts/run_autoencoder_fusion.py`. Mode `resampled` hides a fresh 15% of
  the recorded nonzero entries of the training cells in each epoch and
  recomputes the gene median, SVD and kNN proposals
  (`<methods_root>/autoencoder_fusion_3teachers`). Mode `masked_positives`
  learns from the masked positives with the five teacher contracts
  (`<methods_root>/autoencoder_fusion_5teachers`).
- `slurm_value_accuracy.sh` runs `scripts/evaluate_value_accuracy.py`. It
  keeps the zeros selected by Safe Fusion fixed and inserts each candidate
  value: the boosted fused value, the linear combination, the two autoencoder
  networks and each teacher. It also reads the mask replicates of section 5
  and the thinning units of section 7.

The outputs are in `artifacts/paper_evidence/value_accuracy/`:
`error_removed.csv` (error removed at 1% to 10%, Table 2 and the top of
Table S7), `log_error.csv` (mean absolute log error, Table 2),
`log_error_strata.csv` (by true count and gene detection rate, Table S7),
`cross_validation.csv` (five-fold error of the boosted and linear value
models), `replicates.csv` (mask replicate and thinning units) and
`summary.json` (paired bootstrap intervals).

## 9. Table 4: filling recorded zeros in held-out cells

The deployment analysis applies every method to the recorded counts of the
held-out cells. The training cells keep the benchmark mask, so the selector
still learns from their masked positives. `scripts/deployment_paths.sh`
defines the nine units and `DEPLOY_ROOT` (default
`artifacts/paper_evidence/downstream_deployment`).

```bash
prepare=$(sbatch --parsable scripts/slurm_deployment_prepare.sh)
scvi_dep=$(sbatch --parsable --dependency=afterok:$prepare scripts/slurm_deployment_scvi.sh)
fill=$(sbatch --parsable --dependency=afterok:$scvi_dep scripts/slurm_deployment_fill.sh)
deploy_eval=$(sbatch --parsable --dependency=afterok:$fill scripts/slurm_deployment_evaluate.sh)
```

After `deploy_eval` finishes, run on a compute node:

```bash
python scripts/summarize_fill_evaluations.py \
  --root artifacts/paper_evidence/downstream_deployment/evaluation \
  --output artifacts/paper_evidence/downstream_deployment/summary.csv
```

- `slurm_deployment_prepare.sh` (array 0-8) runs `build_deployment_inputs.py`,
  which writes `hybrid.h5ad` (training cells masked, held-out cells
  recorded), `recorded.h5ad`, `coordinates.parquet`,
  `empty_coordinates.parquet` and `splits.parquet` into each unit directory,
  and fits the gene median, SVD, weighted kNN and MAGIC teachers into
  `methods/`. `slurm_deployment_scvi.sh` fits the scVI teacher.
- `slurm_deployment_fill.sh` fits the fused value and the selector, fills SVD
  and weighted kNN at the same fractions, and runs
  `finalize_deployment_contracts.py`, which writes the evaluated matrices
  (`safe_fusion_<p>pct`, `svd_<p>pct`, `weighted_knn_<p>pct` and the dense
  outputs) with recorded counts outside the held-out cells.
- `slurm_deployment_evaluate.sh` (array 0-8) evaluates them with the
  downstream evaluators of section 13, with the unfilled recorded matrix as
  the reference, and writes
  `artifacts/paper_evidence/downstream_deployment/evaluation/`.

`summary_key_metrics.csv` next to `summary.csv` holds the endpoints of
Table 4.

## 10. Supplementary Table S11: fill-fraction rule

`slurm_detection_rule.sh` refits the selector with the production settings,
calibrates its scores by isotonic regression on cross-fitted scores of the
training cells, and fills every held-out zero whose detection probability
p / (p + ρ(1 − p)) exceeds 1/2, with ρ = 0.10
(`--detection-rule-mask-rate 0.10`). Tasks 0-4 (colon, the pancreas folds,
Norman) apply the rule to the masked benchmark. Tasks 5-9 apply it to the
deployment input of section 9, where every held-out candidate is a recorded
zero.

```bash
sbatch --array=0-4 --dependency=afterok:${stack} scripts/slurm_detection_rule.sh
sbatch --array=5-9 --dependency=afterok:${fill} scripts/slurm_detection_rule.sh
```

The `detection_rule` block of
`artifacts/paper_evidence/detection_rule/<unit>/calibration_report.json`
holds the chosen fill fraction, its masked F1, and the fraction and F1 with
the highest F1 in hindsight. In
`artifacts/paper_evidence/downstream_deployment/{colon,pancreas/fold_<k>,norman_crispra}/detection_rule/calibration_report.json`,
`detection_rule.test_fill_fraction` is the share of recorded zeros that the
rule fills (last column of Table S11).

## 11. Table 3 and Supplementary Table S8: zeros created by knockdown

This analysis ranks the zeros of the knocked-down gene in the recorded counts
of the held-out screen cells (section 9), and reports masked F1 on the four
masked screen benchmarks. `slurm_condition_aware_screens.sh` fits Safe Fusion
with perturbation labels (obs column `target`). Array task = 2 × screen +
setting, with screens Norman, Adamson, Dixit and Papalexi. Even tasks use the
masked benchmark, and odd tasks use the deployment input of section 9.
`STAGE=knn` fits the weighted kNN teacher within each label, `STAGE=scvi` the
scVI teacher with the label as a covariate, and `STAGE=select` the fused value
from the gene median, SVD, MAGIC and the two label teachers, followed by the
selector with the two label features. `STAGE=matched` fills the deployment
zeros at 1% to 10% with the label kNN and scVI teachers and with the MAGIC and
scVI teachers, each ranked by its own value.

```bash
C=scripts/slurm_condition_aware_screens.sh
knn_c=$(sbatch --parsable --array=0-7 --dependency=afterok:${prepare} --export=ALL,STAGE=knn $C)
scvi_c=$(sbatch --parsable --array=0-7 --gres=gpu:l40s:1 --dependency=afterok:${prepare} --export=ALL,STAGE=scvi $C)
select_c=$(sbatch --parsable --array=0-7 --dependency=afterok:${teachers}:${prepare}:${knn_c}:${scvi_c} --export=ALL,STAGE=select $C)
matched_c=$(sbatch --parsable --array=1,3,5,7 --dependency=afterok:${prepare}:${scvi_dep}:${knn_c}:${scvi_c} --export=ALL,STAGE=matched $C)
sbatch --dependency=afterok:${fill}:${select_c}:${matched_c}:${selectors} scripts/slurm_evaluate_perturbation_zeros.sh
```

The standard selectors of the four screens are tasks 3-6 of
`slurm_complete_downstream_selectors.sh` (section 3.2). To run only this
item, submit them with
`sbatch --array=3-6 --dependency=afterok:${stack} scripts/slurm_complete_downstream_selectors.sh`.

The masked setting writes `graph_smooth_condition`,
`scvi_inductive_condition`, `safe_fusion_condition` and `selector_condition`
under `<methods_root>`. The deployment setting writes the same teachers to
`methods/` and the fills to `selector_condition/` and `matched_condition/`
in `artifacts/paper_evidence/downstream_deployment/<screen>/`.
`slurm_evaluate_perturbation_zeros.sh` runs `evaluate_perturbation_zeros.py`
and `evaluate_condition_masked_f1.py` and writes
`artifacts/paper_evidence/perturbation_zeros/`:

- `report.json` and `per_target.csv`: the fill shares of control and
  knocked-down zeros at 1% to 10% and the AUROC with intervals over targets
  (Table 3), and `auroc_by_screen` (Table S8).
- `masked_f1_report.json` and `masked_f1_by_fraction.csv`: the mean masked F1
  over the fill fractions 1% to 10% in each masked screen and averaged over
  the four screens, with and without labels (Masked F1 column of Table 3).

## 12. Figure 2 and Supplementary Table S9: agreement with surface protein

```bash
after=afterok:${audit}:${pdl1_scvi}:${pdl1_selector}
sbatch --dependency=${after} scripts/slurm_evaluate_papalexi_cd274.sh
sbatch --dependency=${after} scripts/slurm_evaluate_papalexi_crossmodal.sh
sbatch --dependency=${after} scripts/slurm_evaluate_papalexi_crossmodal_raw.sh
sbatch --dependency=${after}:${selectors}:${matched} scripts/slurm_compare_method_rankings.sh
```

- `slurm_evaluate_papalexi_cd274.sh` runs `evaluate_papalexi_crossmodal.py`
  for CD274 on centered log ratio and raw PD-L1
  (`benchmark/evaluation_cd274/`, `benchmark/evaluation_cd274_raw_counts/`),
  then `evaluate_pdl1_state_baselines.py`, which adds the target and
  interferon-γ baselines, the paired differences and the partial correlations
  with their intervals (`benchmark/evaluation_pdl1_state/`). It then plots
  Figure 2 with `plot_biological_range_figures.py --pdl1-only` in
  `.venv-baselines`, which writes `pdl1_range_validation.{png,pdf}` to
  `artifacts/paper_evidence/figures/`. `PAPER_DIR=<dir>` writes an extra PNG
  copy to `<dir>`.
- `slurm_evaluate_papalexi_crossmodal.sh` and
  `slurm_evaluate_papalexi_crossmodal_raw.sh` evaluate CD86, PD-L2 and TIM-3
  together with CD274, on centered log ratio and raw counts
  (`benchmark/evaluation/`, `benchmark/evaluation_raw_counts/`).
- `slurm_compare_method_rankings.sh` writes
  `artifacts/paper_evidence/ranking_comparison/`, with the partial Spearman
  correlations of the Safe Fusion and SVD rankings with PD-L1
  (`cd274_report.json`) and the overlap of the zeros filled by Safe Fusion,
  SVD and weighted kNN in the tissues.

The `benchmark/` paths are under
`artifacts/paper_evidence/papalexi_crossmodal/`. In each evaluation
directory, `protein_threshold_range_metrics.csv` holds the threshold AUROC,
`continuous_protein_association.csv` the Spearman correlations,
`fill_range_protein_enrichment.csv` and `range_leaders.csv` the enrichment of
the filled cells, `replicate_association.csv` the correlation within each
replicate, and `permutation_tests.csv` the permutation tests. Table S9 takes
its replicate columns from `evaluation_cd274`, the raw column from
`evaluation_cd274_raw_counts`, and the CD86, PD-L2 and TIM-3 columns from
`evaluation`.

### Figure 2 from the released inputs

`data/papalexi_crossmodal_inputs.tar.gz`, stored with Git LFS, holds the
inputs of these evaluations: the prepared RNA-protein file, the audit panel,
the masked benchmark, the five teacher contracts, the fused value, standard
scVI and the selector scores (see [data/README.md](../data/README.md) for
the checksum and contents). Unpack it at the repository root and run the
evaluations without the chain above:

```bash
git lfs pull --include "data/papalexi_crossmodal_inputs.tar.gz"
tar -xzf data/papalexi_crossmodal_inputs.tar.gz
sbatch scripts/slurm_evaluate_papalexi_cd274.sh
sbatch scripts/slurm_evaluate_papalexi_crossmodal.sh
sbatch scripts/slurm_evaluate_papalexi_crossmodal_raw.sh
```

These launchers need `.venv`, and the Figure 2 plot also needs
`.venv-baselines` (section 1).

## 13. Supplementary Table S10: downstream analyses on the masked benchmark

The downstream benchmark uses the donor-held-out pancreas and colon data, the
held-out zebrafish cells, and the Norman, Adamson, Dixit and Papalexi
interventions. Safe Fusion, SVD and weighted kNN fill 1% to 10% of the
candidate zeros. `MARKER_PANEL` (default `source`, the markers of the source
studies in `src/safefusion_benchmark/marker_panels.py`) selects the marker
panel. [COMPLETE_DOWNSTREAM_PROTOCOL.md](COMPLETE_DOWNSTREAM_PROTOCOL.md)
defines the tasks.

```bash
down_eval=$(sbatch --parsable --dependency=afterok:${selectors}:${matched} scripts/slurm_evaluate_complete_downstream.sh)
sbatch --dependency=afterok:${down_eval} scripts/slurm_finalize_complete_downstream.sh
decompose=$(sbatch --parsable --dependency=afterok:${selectors}:${matched} scripts/slurm_decompose_fills.sh)
```

After `decompose` finishes, run on a compute node:

```bash
python scripts/summarize_fill_evaluations.py \
  --root artifacts/paper_evidence/downstream_decomposition \
  --output artifacts/paper_evidence/downstream_decomposition/summary.csv
```

- `slurm_evaluate_complete_downstream.sh` (array 0-8) writes
  `artifacts/paper_evidence/downstream_complete/{clustering,markers,pancreas_biology,trajectory,grn}/`.
  `slurm_finalize_complete_downstream.sh` writes the combined tables
  (`all_bootstrap_summaries.csv`, the values of Table S10) and the five-panel
  figure to `artifacts/paper_evidence/downstream_complete/summary/`. It fails
  unless all five tasks, all ten fill fractions, all 24 pancreas donors, all
  9 colon donors, all 12 zebrafish stages and all 128 perturbation regulators
  are present and pass the leakage checks.
- `slurm_decompose_fills.sh` (array 0-8, task = 3 × dataset + source, with
  datasets colon, pancreas and zebrafish and sources Safe Fusion, SVD and
  weighted kNN) splits each fill into its selected masked positives and its
  selected recorded zeros (`decompose_fills.py`, which writes
  `decomposition_counts.csv`) and evaluates each part alone. The parts and
  their evaluations are in
  `artifacts/paper_evidence/downstream_decomposition/<dataset>/<source>/`, and
  `summary.csv` and `summary_key_metrics.csv` collect the comparisons with the
  masked input. The Results text on masked positives and recorded zeros comes
  from these files.

## 14. Supplementary Table S1: fill decisions of standard imputers

```bash
jid=$(sbatch --parsable scripts/slurm_standard_imputers.sh)
sbatch --dependency=afterok:${jid} scripts/slurm_evaluate_fill_decisions.sh
```

`slurm_standard_imputers.sh` (array 0-5, one imputer per task: SAUCIE, MAGIC,
DeepImpute, scScope, scVI and kNN smoothing) downloads the CELLxGENE colon
file, checks its SHA-256, creates `.conda-standard-imputers` on first use, and
runs every imputer on CPU with seed 1729 on the four Crohn's disease subsets.
It writes
`artifacts/paper_evidence/standard_imputers/<method>/<dataset>/<disease>/<tissue>.npy`.
`slurm_evaluate_fill_decisions.sh` writes
`artifacts/paper_evidence/fill_decisions/`, where `table_s1.csv` holds
Table S1 and `gene_patterns.csv` and `pair_agreement.csv` hold the per-gene
decisions and the agreement analysis of Supplementary Section S1.
[scripts/standard_imputers/README.md](../scripts/standard_imputers/README.md)
describes the imputer settings and environment.

## 15. Supplementary Table S2: benchmark data

```bash
sbatch --dependency=afterok:${pdl1_prep} scripts/slurm_summarize_benchmark_data.sh
```

The launcher counts the cells, genes, held-out candidates and masked
positives of every unit of `scripts/unit_paths.sh` and writes
`artifacts/paper_evidence/benchmark_data/benchmark_data.csv`. It needs the
inputs of section 2 only.

## 16. Supplementary Table S3 and Section S3: comparison methods

Table S3 lists the settings of the launchers of section 3.3. The ALRA rank is
`chosen_k` in the `metadata.json` of each ALRA contract. The scGCL comparison
of Supplementary Section S3 uses the rows of `scGCL` (comparison
`supplementary`) in the curve files of section 4.

scGCL runs through `scripts/run_scgcl_baseline.py`, an adapter for the
official repository at commit `317015acdf06d2929c20a7d2858bac539b3d8ebd`
(section 1). The adapter keeps the upstream median-library normalization,
Scanpy variable gene selection, the undirected 15-nearest-neighbor graph, the
AFGRL encoder and the summed ZINB objective. It changes only input and output
handling and device placement of tensors. Upstream exports an L2-normalized
log-expression embedding, so the count-scale contract restores the input L2
magnitude, applies `expm1` and keeps the library mass of the selected genes.
At the default learning rate of 1e-3, the pancreas loss becomes nonfinite at
epoch 68. At 1e-4 it fails at epoch 165, and at 1e-5 colon fails after epoch
240. The launchers therefore use 1e-6 for 300 epochs on CPU, with every other
setting unchanged, and record the default, the learning rate used and the
failure note in the contract metadata. The adapter writes a checkpoint every
25 epochs. `sbatch_scgcl_baselines.s` resumes from it after preemption and
runs `scripts/evaluate_scgcl_baseline.py` on each fit. `slurm_scgcl_norman.sh`
fits Norman without the evaluator.

## 17. Supplementary Table S12: runtime

```bash
python3 scripts/summarize_runtime.py
```

`scripts/summarize_runtime.py` reads `scripts/runtime_jobs.tsv`, which lists
the Slurm tasks of the paper run that fitted each step for each dataset
(section 3.2), and queries `sacct` for their elapsed time and peak memory. To
summarize your own run, replace the job IDs in `scripts/runtime_jobs.tsv` with
those of your `teachers`, `scvi`, `stack` and `selectors` jobs.
`--sacct-file` reads a saved pipe-delimited `sacct` dump instead of querying
Slurm. The script writes `runtime_table.csv` (Table S12),
`runtime_by_dataset.csv` and `sacct_records.txt` to
`artifacts/paper_evidence/runtime/`.
