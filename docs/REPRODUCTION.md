# Reproducing the Safe Fusion manuscript

Safe Fusion uses transductive teachers in the main analyses. They use all input
cells without cross-fitting; the fused value and selector still learn from
masked training entries. The inductive mode fits teachers on training cells and
cross-fits their training proposals. Detection-weighted deployment inserts the
fused count multiplied by the calibrated detection probability.

The table below covers Tables 1–3, Figures 1–2, Supplementary Figure S1 and Supplementary Tables S1–S29.
The supplementary labels follow their order in `supplementary_results.tex`.
Supplementary Sections S1–S9 cover fill rates; data; models; recovery; thinning
and values; known zeros; downstream and protein; budgets and scale; and lupus.

Sections 1–18 build the shared inputs, comparator contracts and inductive
reference evaluations. Their reference outputs are prerequisites, not substitutes
for the final transductive results. Sections 19–35 produce the manuscript
analyses listed below.
Read each section's prerequisites before submitting it. Complete its jobs
successfully before starting the next dependent section. Commands containing
`sbatch --wait` wait for successful completion; launchers with `submit` print
job IDs that must finish before continuing. Do not run two copies of a stage.

Saved inputs for the protein evaluations are distributed through Git LFS
(see sections 14 and 22 and [data/README.md](../data/README.md)). Download the
other public inputs using [DATA.md](DATA.md) and the fetch commands below.
Run all commands from this repository's root. Never link the output tree to
another run that must remain unchanged.

| Manuscript item | Content | Section | Output files |
|---|---|---|---|
| Table 1 | Masked recovery | 21, 27, 29 | `R4/transductive_comparators/paired_differences.csv`; `R4/transductive_main/masked/<dataset>/{paired_differences,new_comparators_paired_differences}.csv`; `R4/round2_extras/part_b/highest_f1.csv` |
| Table 2 | Zeros with known status | 22, 23, 28, 29 | `R4/round2_extras/part_a/table2_aurocs.csv` (continuous estimates and intervals, including label-aware rows) |
| Table 3 | Error against deeper counts | 23, 31 | `R4/reviewer_extras/B_thinning/{all_rows,table3_new_rows_x100}.csv` |
| Figure 1 | Overview and masked F1 | 21, 27, 29 | `R4/round2_extras/part_b/selector_f1_fillrate_allcell_1000_points.csv`; `R4/transductive_main/figures/f1_fillrate_3panel_oup.png` |
| Figure 2 | CD274 RNA and PD-L1 protein | 22 | `R4/transductive_references/protein_cs/figures/pdl1_range_validation.png` |
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
| Supplementary Table S21 | Masked downstream endpoints | 23 | `R4/transductive_downstream/summary/{s17_absolute,endpoint_changes,decomposition_counts}.csv` |
| Supplementary Table S22 | Recorded-count downstream endpoints | 23 | `R4/transductive_downstream/summary/endpoint_changes.csv` |
| Supplementary Table S23 | Disease effects and null tests | 23 | `R4/transductive_downstream/summary/disease_all.csv` |
| Supplementary Table S24 | Reference mapping changes | 23 | `R4/transductive_downstream/disease/<tissue>_<fold>/deployment/annotation/<tissue>/overall.csv` |
| Supplementary Table S25 | Correlation null | 23 | `R4/transductive_downstream/correlation/null/counts/{summary,versus_reference}.csv` |
| Supplementary Table S26 | Fill-fraction rule | 30 | `R4/current_tables/current/rule/summary.csv`; `current/rule/{masked,recorded}/<unit>/calibration_report.json` |
| Supplementary Table S27 | Runtime | 18 | `runtime/runtime_table.csv` |
| Supplementary Table S28 | Scaling study | 26, 31, 33, 34 | `R3/scale/results/{paired_differences,resources,safe_fusion_totals}.csv`; Section S8 extensions: `R4/reviewer_extras/C_scale/results/paired_differences.csv`, `R4/scale_caps/genes_<2000 or 5000>/evaluation/results/paired_differences.csv`, `R4/scale_comparators/{paired_differences,resources}.csv` |
| Supplementary Table S29 | Lupus Treg example | 24, 35 | `R4/sle_donor_labels/comparisons/masked_f1_intervals.csv`; `comparisons/{q1_zeros_sex/per_gene,q3_clustering/treg_calls_summary,q3_clustering/clustering_mean_over_folds,focus_fills/focus_fill_counts,deploy_main/q5_modules,deploy_main/q4_effects,deploy_main/q4_null_de,q7_protein/fcrl3_protein_agreement}.csv` |

Output abbreviations: `R2`, `R3`, `R4` denote `review_round2`, `review_round3`,
`review_round4` under `artifacts/paper_evidence/`.

## Conventions

- Run every command from the repository root. Each launcher changes to
  `SLURM_SUBMIT_DIR` and writes its log to `logs/`, so create that
  directory first (`mkdir -p logs`).
- All compute runs through Slurm. Most launchers carry
  `#SBATCH --account=torch_pr_634_general`, the account of the cluster where
  the paper was run. Set your own account (and partition, if your cluster
  needs one) once per shell. Slurm input environment variables override
  `#SBATCH` lines, and they also apply to the `sbatch` calls inside the
  launchers that have a `submit` stage:

  ```bash
  export SBATCH_ACCOUNT=<account>
  export SBATCH_PARTITION=<partition>   # only if your cluster needs it
  ```

- The selector scores depend on the CPU model, because the BLAS kernels of
  the multilayer perceptron differ between processors. The paper ran the
  selectors on the `cs` partition of its cluster, whose nodes have Intel Xeon
  Platinum 8592+ processors. `slurm_fusion_value.sh` and
  `slurm_evaluate_protein_within_state.sh` request that partition with
  `#SBATCH --partition`. On another processor model the selector scores, the
  selected zeros and every number that depends on them change slightly. Run
  all selector fits on one processor model.
- Every scVI fit requests an NVIDIA L40S GPU (`--gres=gpu:l40s:1`), because
  scVI outputs differ between GPU types. The scGPT launchers request an NVIDIA
  H200 (`--gres=gpu:h200:1`).
- Every ALRA fit fixes `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `MKL_NUM_THREADS` at 8, because the randomized SVD of ALRA depends on the
  number of BLAS threads.
- The launchers and scripts fix their seeds. Most splits, masks, models and
  bootstraps use seed 1729. The mask replicates use seeds 1730 to 1733, the
  thinning bootstraps use seed 7, and scGCL and Mixscape keep their upstream
  seed 0. The package versions are pinned in `uv.lock`, `workflow/envs/`,
  `scripts/standard_imputers/environment.yaml` and the setup launchers of
  section 1. Numerical results can still vary across hardware and library
  builds.
- Each section stores job IDs in shell variables (`teachers`, `lf_selector`,
  ...), and later sections depend on them. Submit the sections in order from
  one shell, or drop a dependency on a job that has already finished.
- Where the paper's jobs requested more memory or time than a launcher's
  `#SBATCH` lines, the commands below pass `--mem` or `--time`.
- Python commands that have no launcher also run on a compute node, for
  example inside `srun`, with the environment of section 1
  (`source .venv/bin/activate`).
- `artifacts/`, `external_data/`, `logs/` and the environments are ignored by
  Git.

### Output directories

Two benchmark builds feed the manuscript.

- The production build, under `artifacts/paper_evidence/`, holds colon, the
  pancreas benchmark with one set of 1,208 genes, the Adamson, Dixit and
  Papalexi screens, the zebrafish time course and the Papalexi RNA-protein
  benchmark.
- The leakage-free build, under
  `artifacts/paper_evidence/review_round2/leakage_free/`, holds the pancreas
  donor folds with variable genes selected within each fold and the Norman
  CRISPR activation screen with conditions selected on development cells.
  This build supplies the earlier inductive masked-recovery, selector,
  thinning and value comparisons, Supplementary Table S11, the pancreas fold
  rows of Supplementary Table S2 and the Norman analyses. The Norman analyses
  beyond masked recovery write to
  `artifacts/paper_evidence/review_round2/norman_rebuilt/`.

Additional reference analyses write to
`artifacts/paper_evidence/review_round2/<analysis>/` and
`artifacts/paper_evidence/disease_control_checks/`. The launchers read the
colon and pancreas workflow runs of section 2.2 from the fixed directories
`artifacts/colon_runs/0b2469810675-c0db6f963e94` and
`artifacts/pancreas_runs/0b2469810675-45c81b160d78`.

## 1. Environments

```bash
uv sync --python 3.11 --extra workflow --extra paper
source .venv/bin/activate
mkdir -p logs

sbatch scripts/slurm_setup_current_baselines.sh    # .conda-scvi-current and .conda-magic-current
sbatch scripts/sbatch_setup_r_baselines.s          # .conda-r-baselines
sbatch scripts/sbatch_setup_python_baselines.s     # .venv-baselines and the scGPT checkpoint
sbatch scripts/sbatch_setup_pertpy_env.s           # .venv-pertpy

git clone https://github.com/zehaoxiong123/scGCL.git baselines_and_data/scGCL
git -C baselines_and_data/scGCL checkout 317015acdf06d2929c20a7d2858bac539b3d8ebd
```

Python 3.12 also works for `.venv`. The environments run the following
steps.

| Environment | Built from | Runs |
|---|---|---|
| `.venv` | `pyproject.toml`, `uv.lock` | Workflow, gene median, SVD and weighted kNN teachers, fused value, selector, ALRA (Python port), SAVER driver, autoencoder fusion network, evaluations and inductive F1 curves |
| `.conda-scvi-current` | `workflow/envs/scvi.yaml` | scVI teacher, standard scVI, the probability of a nonzero count (`nonzero_probability.py`), Papalexi RNA-protein preparation and audit, label audit |
| `.conda-magic-current` | `workflow/envs/magic.yaml` | MAGIC teacher, standard MAGIC |
| `.conda-r-baselines` | `scripts/sbatch_setup_r_baselines.s` | SAVER (`scripts/run_saver_baseline.R`, called by `run_saver_baseline.py --rscript`) |
| `.venv-baselines` | `scripts/sbatch_setup_python_baselines.s` | scGPT (checkpoint in `external_data/baselines/scgpt_human`), the scGCL adapter (which also imports `scanpy` and `faiss`), and the protein plots |
| `.venv-pertpy` | `scripts/sbatch_setup_pertpy_env.s` | Mixscape classes of the Papalexi cells (pertpy 1.3.0, scanpy 1.12.4, mudata 0.4.1) |
| `.venv-scanpy` | `scripts/slurm_disease_control_analysis.sh` on first use | Module scores and Leiden clusters of section 12 (scanpy 1.11.5, leidenalg 0.12.0, igraph 1.0.0 on the numerical stack of `.venv`) |
| `.conda-standard-imputers` | `scripts/standard_imputers/environment.yaml` | The six imputers of Supplementary Table S1, created by `slurm_standard_imputers.sh` on first use |

## 2. Data preparation

[DATA.md](DATA.md) lists the public sources and the SHA-256 of every input.
Run the commands of sections 2.1 to 2.7 on a compute node.

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
(`--variable-genes`), plus a fixed marker list. Without
`--gene-selection-splits`, the pancreas script selects the variable genes on
the development donors of its own donor assignment. This gives the production
pancreas gene set of 1,208 genes. Section 2.5 selects them within each donor
fold instead.

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
datasets) and the colon `thinning_050` and `thinning_025` of section 8. The
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
its raw UMI counts layer. Task 1 of the `data` stage of
`scripts/slurm_leakage_free_rerun.sh` prepares the benchmark:

```bash
LF=scripts/slurm_leakage_free_rerun.sh
lf_norman=$(sbatch --parsable --array=1 --mem=32G \
  --export=ALL,STAGE=data,NORMAN_SOURCE=<path to perturb_processed.h5ad> $LF)
```

The task runs

```bash
python scripts/prepare_norman_crispra.py --input <perturb_processed.h5ad> \
  --output artifacts/paper_evidence/review_round2/leakage_free/norman_crispra/prepared.h5ad \
  --seed 1729 --max-cells-per-condition 60 --max-control-cells 1000 \
  --min-cells-per-condition 60 --target-log2fc-min 0.5 --target-p-max 0.05 --test-fraction 0.30
python scripts/setup_norman_crispra_experiment.py \
  --input artifacts/paper_evidence/review_round2/leakage_free/norman_crispra/prepared.h5ad \
  --output-dir artifacts/paper_evidence/review_round2/leakage_free/norman_crispra \
  --split-column preassigned_split --mask-fraction 0.10 --seed 1729
```

`prepare_norman_crispra.py` reads the source with bounded h5py block reads,
draws 60 cells of each single-gene condition that has at least 60 cells and
1,000 control cells, and assigns 30% of the cells of each condition to the
test split. It then keeps a condition when, in the development cells, the
mean count of its target gene exceeds that of the control cells by at least
0.5 on the log2(1 + x) scale with a two-sided Mann-Whitney p < 0.05. The
cell labels come from the source rows of the drawn cells. The split is stored
in the obs column `preassigned_split`, and `setup_norman_crispra_experiment.py`
must be called with `--split-column preassigned_split` so that the benchmark
keeps the split on which the conditions were selected. The prepared file has
SHA-256 `8aa5b1e3d82a1c57e42e2f4397a37bad2f5181baf04ffca6b161d25a0ad47170`
and holds 65 of 104 single-gene conditions, 1,000 control cells and 5,045
genes (`prepared.report.json`). The benchmark directory also holds
`corrupted.h5ad`, `coordinates.parquet`, `splits.parquet` and
`manifest.json`.

### 2.5 Pancreatic-islet folds with variable genes selected within each fold

```bash
lf_pancreas=$(sbatch --parsable --array=0 --mem=32G --export=ALL,STAGE=data $LF)
```

Task 0 of the `data` stage rebuilds the donor folds of section 2.3 under
`artifacts/paper_evidence/review_round2/leakage_free/pancreas_crossfit/` (the
folds depend only on donors, conditions and the seed, so they equal the
production folds). For each fold it prepares the pancreas file with
`prepare_pancreas_atlas.py --gene-selection-splits fold_<k>/splits.parquet`,
which selects the variable genes on the development donors of that fold
(1,207, 1,209 and 1,207 genes), and masks 10% of the nonzero entries of the
fold within 16 strata of library size and gene mean count
(`make_mask_replicate.py`, seed 1729). It writes
`fold_<k>/{prepared.h5ad,prepared.report.json,corrupted.h5ad,coordinates.parquet,splits.parquet}`.
It then writes `units_manifest.json`, the masked-recovery units of the
leakage-free build (`write_leakage_free_units.py`), in which colon keeps its
production unit, and links the production colon selector and stacked
selectors of sections 3.2 and 4 into the leakage-free directory.

### 2.6 Adamson, Dixit and Papalexi screens and the zebrafish time course

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

### 2.7 Papalexi RNA-protein benchmark

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
  `artifacts/paper_evidence/papalexi_crossmodal/audit/matched_rna_adt_panel.parquet`
  and `audit_report.json` (20,729 cells, 285 barcodes shared by more than one
  cell, 20,156 matched cells; Supplementary Section S10).

### 2.8 Label audit

`scripts/audit_prepared_labels.py` locates the source row of every prepared
cell and reports the share of cells whose counts and labels equal the source,
and the change of each perturbation target gene in the cells labelled with
that target. It needs the sources of sections 2.1, 2.4 and 2.6, the Papalexi
files of section 2.7 and `mudata`, so it runs in `.conda-scvi-current`:

```bash
sbatch --mem=96G -c 4 --time=01:00:00 --dependency=afterok:${lf_norman}:${audit} \
  --wrap ".conda-scvi-current/bin/python scripts/audit_prepared_labels.py \
  --norman-source <path to perturb_processed.h5ad>"
```

It writes `artifacts/paper_evidence/review_round2/label_audit/label_audit.json`.
The Dixit entry gives the median log2 fold change of the knocked-out targets,
-0.009 (Supplementary Section S7), and the Norman entry confirms that every
prepared Norman cell carries its source condition.

## 3. Safe Fusion and the comparison methods

### 3.1 Dataset units

`scripts/unit_paths.sh` defines the nine production units of the launchers,
in array order: `pancreas_0`, `pancreas_1`, `pancreas_2` (the donor folds),
`colon`, `adamson_crispri`, `dixit_ko`, `papalexi_eccite`, `zebrafish` and
`papalexi_crossmodal`. `unit_paths <key>` sets `input`, `coordinates`,
`splits`, `truth` and `methods_root`, and `teacher_contract_args <root>`
expands the five teacher contracts under `<root>`. The key `norman_crispra`
points to the Norman benchmark of section 2.4, which the leakage-free
launcher of section 3.4 fits. A contract is a directory with `mean.npy`
(cells by genes) and `metadata.json`.

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
linear combination used in the value comparisons of section 9 instead. The
contract metadata records the five-fold cross-validated error of both value models
(`parameters.value_model_cross_validation`).

`scripts/calibrated_selective_fill.py --architecture mlp --budget-mode
apply_topk` fits the MLP selector on the selector-fitting cells (the
validation donors in colon, the development cells elsewhere) and fills the
top-ranked zeros of the held-out cells with the fused value at each fill
fraction. It writes one `safe_fusion_calibrated_mlp_topk_<b>` contract per
fraction, where `<b>` is the fraction with `p` in place of the decimal point
(`0p01` to `0p1`), and `calibration_report.json` with the 1000-point masked
F1 curve from 0.1% to 100%. With `--condition-column target` (section 13),
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

- `slurm_prepare_complete_downstream_teachers.sh` (array 0-35, task =
  4 × unit + teacher) fits the gene median, SVD, weighted kNN and MAGIC
  teachers of every unit. `slurm_scvi_teachers.sh` (array 0-8) fits the scVI
  teacher. The contracts are
  `<methods_root>/{gene_median,svd_impute,graph_smooth,magic_inductive,scvi_inductive}`.
- `slurm_safe_fusion_stack.sh` (array 0-8) writes the fused value to
  `<methods_root>/safe_fusion`. With `VALUE_MODEL=linear` it writes
  `<methods_root>/safe_fusion_linear` (section 9).
- `slurm_complete_downstream_selectors.sh` (array 0-7, in the order
  `pancreas_0`, `pancreas_1`, `pancreas_2`, `adamson_crispri`, `dixit_ko`,
  `papalexi_eccite`, `zebrafish`, `colon`) fits the selector and writes the
  fills at 1% to 10%. The output directories are
  `artifacts/paper_evidence/selector_mlp_biology_range/colon`,
  `artifacts/paper_evidence/pancreas_crossfit/fold_<k>/selector_mlp_biology_range_fullteachers`
  and `<methods_root>/selector_mlp_biology_range` for the three screens and
  zebrafish.
- `slurm_apply_fill_fraction.sh` (array 0-7) fills SVD and weighted kNN at
  the same fractions, each ranking zeros by its own value
  (`scripts/apply_fill_fraction.py`), and writes `svd_topk_<b>` and
  `weighted_knn_topk_<b>` under
  `artifacts/paper_evidence/matched_fraction/colon`,
  `artifacts/paper_evidence/pancreas_crossfit/fold_<k>/matched_fraction` and
  `<methods_root>/matched_fraction` for the screens and zebrafish.
- `slurm_papalexi_crossmodal_mlp.sh` fits the selector of the RNA-protein
  benchmark and writes
  `artifacts/paper_evidence/papalexi_crossmodal/benchmark/mlp_selector/`,
  including `selected_gene_scores.parquet` with the scores of the zeros of
  CD274, CD86, PDCD1LG2 and HAVCR2.
- `slurm_papalexi_crossmodal_scvi.sh` fits standard scVI on the RNA-protein
  benchmark (`benchmark/scvi`).

### 3.3 Comparison methods on the production tissues

```bash
alra_colon=$(sbatch --parsable scripts/slurm_alra_colon.sh)
alra_pancreas=$(sbatch --parsable scripts/slurm_alra_pancreas_crossfit.sh)
saver_tissue=$(sbatch --parsable scripts/sbatch_run_saver_pc.s)
saver_rescale=$(sbatch --parsable --dependency=afterok:${saver_tissue} scripts/slurm_rescale_saver_contracts.sh)
magic_std=$(sbatch --parsable scripts/slurm_magic_current_baselines.sh)
scvi_std=$(sbatch --parsable scripts/slurm_scvi_current_baselines.sh)
scgpt=$(sbatch --parsable scripts/slurm_scgpt_gene_current_baselines.sh)
scgcl=$(sbatch --parsable scripts/sbatch_scgcl_baselines.s)
baselines=${alra_colon}:${alra_pancreas}:${saver_rescale}:${magic_std}:${scvi_std}:${scgpt}:${scgcl}
```

| Launcher | Method | Output |
|---|---|---|
| `slurm_alra_colon.sh` | ALRA, colon | `artifacts/paper_evidence/baselines/alra/colon_mask_010` |
| `slurm_alra_pancreas_crossfit.sh` (array 0-2) | ALRA, pancreas folds | `artifacts/paper_evidence/pancreas_crossfit/fold_<k>/alra` |
| `sbatch_run_saver_pc.s` | SAVER, pancreas and colon | `artifacts/paper_evidence/baselines/saver/{pancreas,colon}_mask_010` |
| `slurm_magic_current_baselines.sh` (array 0-1) | Standard MAGIC | `artifacts/paper_evidence/baselines/magic/{pancreas,colon}` |
| `slurm_scvi_current_baselines.sh` (array 0-1) | Standard scVI | `artifacts/paper_evidence/baselines/scvi/{pancreas,colon}` |
| `slurm_scgpt_gene_current_baselines.sh` (array 0-1) | scGPT masked value decoder | `artifacts/paper_evidence/baselines/scgpt_mvc/{pancreas,colon}` |
| `sbatch_scgcl_baselines.s` (array 0-1) | scGCL, pancreas and colon | `artifacts/paper_evidence/baselines/scgcl/{pancreas,colon}` |

ALRA is fitted on the training cells and applied to the held-out cells
(`scripts/run_alra_baseline.py`, a Python port of ALRA). SAVER, MAGIC, scVI
and scGCL are fitted on the whole masked matrix. `run_saver_baseline.py`
writes SAVER on the count scale and marks the contract
`count_scale_rescaled`. `slurm_rescale_saver_contracts.sh` puts SAVER
contracts that lack this mark on the count scale and skips the others.
`scripts/masked_f1_units.py` defines the evaluation units, the comparator
contract paths and the conversion of each contract to the count scale.

### 3.4 Leakage-free build: pancreas folds and the Norman screen

`scripts/slurm_leakage_free_rerun.sh` fits the five teachers, the fused
value, the selector and every comparison method on the four units of the
leakage-free build, in array order `pancreas_0`, `pancreas_1`, `pancreas_2`
(the folds of section 2.5) and `norman_crispra` (section 2.4), with the
production settings and seeds. `STAGE` selects the step.

```bash
after_data=afterok:${lf_norman}:${lf_pancreas}
lf_teachers=$(sbatch --parsable --array=0-15 --mem=32G --dependency=${after_data} --export=ALL,STAGE=teachers $LF)
lf_scvi_teacher=$(sbatch --parsable --array=0-3 --mem=32G --gres=gpu:l40s:1 --dependency=${after_data} --export=ALL,STAGE=scvi_teacher $LF)
lf_stack=$(sbatch --parsable --array=0-3 --dependency=afterok:${lf_teachers}:${lf_scvi_teacher} --export=ALL,STAGE=stack $LF)
lf_selector=$(sbatch --parsable --array=0-3 --mem=32G --dependency=afterok:${lf_stack} --export=ALL,STAGE=selector $LF)
lf_alra=$(sbatch --parsable --array=0-3 --dependency=${after_data} --export=ALL,STAGE=alra $LF)
lf_saver=$(sbatch --parsable --array=0-2 --time=03:00:00 --dependency=${after_data} --export=ALL,STAGE=saver $LF)
lf_saver_norman=$(sbatch --parsable --array=3 --mem=120G --time=06:00:00 --dependency=${after_data} --export=ALL,STAGE=saver $LF)
lf_magic=$(sbatch --parsable --array=0-3 --mem=96G --dependency=${after_data} --export=ALL,STAGE=magic $LF)
lf_scvi=$(sbatch --parsable --array=0-3 --mem=64G --gres=gpu:l40s:1 --dependency=${after_data} --export=ALL,STAGE=scvi $LF)
lf_scgpt=$(sbatch --parsable --array=0-3 --mem=64G --gres=gpu:h200:1 --dependency=${after_data} --export=ALL,STAGE=scgpt $LF)
lf_baselines=${lf_alra}:${lf_saver}:${lf_saver_norman}:${lf_magic}:${lf_scvi}:${lf_scgpt}
```

- `teachers` (task = 4 × unit + teacher) and `scvi_teacher` fit the five
  teachers, `stack` the fused value and `selector` the MLP selector with the
  1000-point masked F1 curve. The pancreas contracts are in
  `leakage_free/pancreas_crossfit/fold_<k>/`, with the selector in
  `fold_<k>/selector_mlp_biology_range_fullteachers/`. The Norman contracts
  are in `leakage_free/norman_crispra/methods/`, with the selector in
  `leakage_free/selector_mlp_biology_range_fullteachers/norman_crispra/`.
- `alra`, `saver`, `magic`, `scvi` and `scgpt` fit the comparison methods with
  the settings of section 3.3 and write
  `leakage_free/baselines/<method>/{pancreas_fold_<k>,norman}`.

The directory `leakage_free/` stands for
`artifacts/paper_evidence/review_round2/leakage_free/` here and below.

## 4. Earlier inductive masked-recovery analysis

```bash
stacked=$(sbatch --parsable --dependency=afterok:${teachers}:${scvi}:${baselines} scripts/slurm_stacked_selector_baselines.sh)
curves=$(sbatch --parsable --dependency=afterok:${selectors}:${stacked}:${baselines} scripts/slurm_matched_baseline_curves.sh)
lf_stacked=$(sbatch --parsable --array=0-3 --dependency=afterok:${lf_selector}:${lf_baselines} --export=ALL,STAGE=stacked $LF)
lf_evaluate=$(sbatch --parsable --array=0 --mem=64G \
  --dependency=afterok:${lf_selector}:${lf_stacked}:${lf_baselines}:${curves} --export=ALL,STAGE=evaluate $LF)
```

- `slurm_stacked_selector_baselines.sh` (array 0-3: `pancreas_0`,
  `pancreas_1`, `pancreas_2`, `colon`) trains the Safe Fusion selector on one
  comparator's value plus the three context features, for every comparator
  with values on the selector-fitting cells, and writes
  `artifacts/paper_evidence/stacked_selector_baselines/<unit>/<comparator>/`
  (`report.json`, `unit_counts.parquet`). The `stacked` stage of the
  leakage-free launcher does the same for its four units in
  `leakage_free/stacked_selector_baselines/`, which also links the colon
  results.
- `slurm_matched_baseline_curves.sh` computes the masked F1 curves and paired
  bootstraps of the production tissue units
  (`compute_matched_baseline_f1_curves.py`, `combine_mlp_baseline_curves.py`,
  `paired_masked_f1_bootstrap.py`) under `artifacts/paper_evidence/`. The
  comparison with the leakage-free build and the scGCL comparison of
  Supplementary Section S3 read these files.
- The `evaluate` stage runs the same three scripts on
  `leakage_free/units_manifest.json` (pancreas folds, colon, Norman), plots
  inductive F1 curves with `plot_selector_f1_fillrate.py`, and runs
  `compare_leakage_free_benchmarks.py`.

The outputs are in `leakage_free/`:

- `masked_f1_paired_bootstrap.csv` and `masked_f1_paired_bootstrap.json`: Safe
  Fusion minus each comparison and each stacked selector, at 1%, 2%, 5% and
  10% and averaged over the fill fractions 1% to 10%
  (`mean_difference_1_to_10`), with intervals from 2000 draws of donors or
  perturbation targets. The comparison rows report the earlier inductive
  masked-recovery comparisons. Rows marked `<method> (stacked)` report the
  selector comparisons.
- `selector_f1_fillrate_mlp_baselines_summary.json`: in
  `<dataset>/main/0.001_0.100/fraction_mlp_best`, the share of the 100 fill
  fractions from 0.1% to 10% at which Safe Fusion has the highest F1 in the
  earlier inductive analysis.
- `selector_f1_fillrate_baselines_1000_points.csv`,
  `selector_f1_fillrate_mlp_baselines_1000_points.csv` and
  `masked_f1_unit_counts.parquet`: the curves and per-unit counts.
- Inductive F1 curves: `figures/f1_fillrate_3panel_oup.png`.
- `table1_old_vs_new.csv`: the inductive masked-recovery results for the
  production and leakage-free pancreas benchmarks. Its `change_pp` column
  gives the effect of the gene set on each pancreas difference (Methods, Datasets and splits).
  `leakage_audit.json` compares the gene sets and the mask strata.

## 5. Inductive and transductive teacher fits

`scripts/slurm_fusion_value.sh` refits the teacher and classifier variants
on the units of `leakage_free/units_manifest.json`
(`pancreas_0` to `pancreas_2`, `colon`, `norman_crispra`). It requests the
`cs` partition, on which these variants reproduce the production selectors
entry for entry.

```bash
FV=scripts/slurm_fusion_value.sh
fv_teachers=$(sbatch --parsable --array=0-24 --dependency=afterok:${lf_evaluate} --export=ALL,STAGE=teachers $FV)
fv_stack=$(sbatch --parsable --array=0-4 --dependency=afterok:${fv_teachers} --export=ALL,STAGE=stack $FV)
fv_selectors=$(sbatch --parsable --array=0-89 --dependency=afterok:${lf_evaluate} --export=ALL,STAGE=selectors $FV)
fv_transductive=$(sbatch --parsable --array=0-4 --dependency=afterok:${fv_stack} \
  --export=ALL,STAGE=selectors,FAMILIES=transductive $FV)
fv_probability=$(sbatch --parsable --array=0-9 --dependency=afterok:${lf_evaluate} --export=ALL,STAGE=probability $FV)
fusion_value=$(sbatch --parsable --dependency=afterok:${fv_selectors}:${fv_transductive}:${fv_probability} \
  --export=ALL,STAGE=evaluate $FV)
```

- `teachers` (task = 5 × unit + teacher) fits the transductive teachers: the
  gene median, SVD and weighted kNN on the masked counts of all cells
  (`run_leakage_safe_method.py --transductive`), and count-scale copies of the
  unit's standard MAGIC and scVI (`count_scale_contract.py`). `stack` fits the
  fused value on them.
- `selectors` runs `fusion_value_selectors.py` for 18 variants per unit: Safe
  Fusion, Safe Fusion without each teacher, the logistic regression and
  gradient-boosted classifiers on the same eight features, and the selector
  trained on one method's value. With `FAMILIES=transductive` it fits the
  transductive variant.
- `probability` (task = 2 × unit + method) writes the probability of a
  nonzero count under the standard scVI fit and under the SAVER posterior
  (`nonzero_probability.py`).
- `evaluate` runs `fusion_value_bootstrap.py`.

The outputs are in
`artifacts/paper_evidence/review_round2/fusion_value/evaluation/`:
`summary_table.csv` holds the teacher and classifier comparisons, including
the scVI probability and transductive variants, with
`paired_differences.csv`, `absolute.csv` and `summary.json` (intervals and
average precision).

## 6. Inductive mask replicates

Each replicate draws a new stratified 10% mask (`make_mask_replicate.py`) and
refits the five teachers, the fused value, the selector, and standard MAGIC
and scVI with seeds 1730 to 1733. Seed 1729 is the run of sections 3 and 4.
The pancreas replicates reuse the three donor folds and the production gene
set. `STAGE` selects the step.

```bash
SR=scripts/slurm_seed_replicates.sh
mask_rep=$(sbatch --parsable --array=0-7 --export=ALL,STAGE=mask $SR)
teachers_rep=$(sbatch --parsable --array=0-63 --dependency=afterok:${mask_rep} --export=ALL,STAGE=teachers $SR)
scvi_teacher_rep=$(sbatch --parsable --array=0-15 --gres=gpu:l40s:1 --dependency=afterok:${mask_rep} --export=ALL,STAGE=scvi_teacher $SR)
stack_rep=$(sbatch --parsable --array=0-15 --dependency=afterok:${teachers_rep}:${scvi_teacher_rep} --export=ALL,STAGE=stack $SR)
rep_lin=$(sbatch --parsable --array=0-15 --dependency=afterok:${stack_rep} --export=ALL,STAGE=stack,VALUE_MODEL=linear $SR)
selector_rep=$(sbatch --parsable --array=0-15 --dependency=afterok:${stack_rep} --export=ALL,STAGE=selector $SR)
magic_rep=$(sbatch --parsable --array=0-7 --dependency=afterok:${mask_rep} --export=ALL,STAGE=magic $SR)
scvi_rep=$(sbatch --parsable --array=0-7 --gres=gpu:l40s:1 --dependency=afterok:${mask_rep} --export=ALL,STAGE=scvi $SR)

NR=scripts/slurm_norman_rebuilt_replicates.sh
nr_mask=$(sbatch --parsable --array=0-3 --mem=16G --dependency=afterok:${lf_norman} --export=ALL,STAGE=mask $NR)
nr_teachers=$(sbatch --parsable --array=0-15 --mem=32G --dependency=afterok:${nr_mask} --export=ALL,STAGE=teachers $NR)
nr_scvi_teacher=$(sbatch --parsable --array=0-3 --mem=32G --gres=gpu:l40s:1 --dependency=afterok:${nr_mask} --export=ALL,STAGE=scvi_teacher $NR)
nr_stack=$(sbatch --parsable --array=0-3 --mem=48G --dependency=afterok:${nr_teachers}:${nr_scvi_teacher} --export=ALL,STAGE=stack $NR)
nr_lin=$(sbatch --parsable --array=0-3 --mem=48G --dependency=afterok:${nr_stack} --export=ALL,STAGE=stack,VALUE_MODEL=linear $NR)
nr_selector=$(sbatch --parsable --array=0-3 --mem=32G --dependency=afterok:${nr_stack} --export=ALL,STAGE=selector $NR)
nr_magic=$(sbatch --parsable --array=0-3 --mem=96G --dependency=afterok:${nr_mask} --export=ALL,STAGE=magic $NR)
nr_scvi=$(sbatch --parsable --array=0-3 --mem=64G --gres=gpu:l40s:1 --dependency=afterok:${nr_mask} --export=ALL,STAGE=scvi $NR)
sbatch --array=0 --mem=32G --dependency=afterok:${nr_selector}:${nr_magic}:${nr_scvi}:${lf_evaluate} --export=ALL,STAGE=summary $NR
```

`slurm_seed_replicates.sh` covers the two tissues (mask, magic and scvi: task
= 2 × seed + dataset; teachers: task = 16 × seed + 4 × unit + teacher; the
other stages: task = 4 × seed + unit). `slurm_norman_rebuilt_replicates.sh`
covers the Norman screen, with seed 1729 from section 3.4. `rep_lin` and
`nr_lin` write the linear combination of each replicate
(`safe_fusion_linear`) for section 9. After the tissue jobs finish, run on a
compute node:

```bash
python scripts/summarize_seed_replicates.py --datasets Pancreas Colon
```

The tissue columns of the upper block come from
`artifacts/paper_evidence/seed_replicates/summary/`
(`seed_replicate_summary.csv`, `seed_replicate_across.csv`,
`seed_replicate_f1_curves.csv`), and the CRISPRa column from the same files in
`artifacts/paper_evidence/review_round2/norman_rebuilt/seed_replicates/summary/`.

## 7. Inductive feature comparisons

```bash
ablation=$(sbatch --parsable --dependency=afterok:${stack} scripts/slurm_selector_mlp_attribution_range.sh)
sbatch --dependency=afterok:${ablation} scripts/slurm_pair_mlp_attribution_range.sh
NS=scripts/slurm_norman_rebuilt_supplement.sh
sbatch --dependency=afterok:${lf_stack} --export=ALL,STAGE=attribution $NS
```

`slurm_selector_mlp_attribution_range.sh` (array 0-3: the pancreas folds,
colon) refits the MLP selector with all features, the teacher features only
and the context features only (`VARIANTS`, default
`full teacher_only context_only`) and fills 1% to 10% of the zeros.
`slurm_pair_mlp_attribution_range.sh` (array 0-1) pools the pancreas folds
and compares every variant with the full selector. The `attribution` stage
of `slurm_norman_rebuilt_supplement.sh` does both for the Norman screen. In
`ranking_metrics.parquet` of
`artifacts/paper_evidence/selector_mlp_attribution_range/{pancreas/fold_<k>,colon}/`
and `artifacts/paper_evidence/review_round2/norman_rebuilt/selector_mlp_attribution_range/norman_crispra/`,
the column `test_pr_auc` holds the test precision-recall AUC of each variant,
next to the masked positive counts. The pancreas values are the means over
the three folds.

## 8. Thinning inputs and inductive evaluation

### 8.1 Thinning benchmark with thinning-trained selectors

The positives are held-out entries that are nonzero in the recorded counts
and zero after thinning. Colon uses the workflow corruptions `thinning_050`
and `thinning_025` (section 2.2), and `prepare_thinning_benchmark.py` thins
pancreas to 50% of the molecules. Every model is refitted on the thinned
training cells, the selector learns from the thinned-out entries of its
fitting cells, and standard MAGIC and scVI are fitted on all cells as
comparators. The `submit` stage calls `sbatch` for the stages `prepare`,
`cpu`, `gpu` (on an L40S), `selector` and `evaluate` with their dependencies
and prints their job IDs. The `stack` stage with `VALUE_MODEL=linear` writes
the linear combination of section 9.

```bash
read -r thin_prep thin_cpu thin_gpu thin_sel thin_eval <<< \
  "$(bash scripts/slurm_thinning_benchmark.sh submit | sed 's/[a-z]*=//g')"
thin_lin=$(sbatch --parsable --job-name=sf-thin-stack --array=0-4 \
  --dependency=afterok:${thin_cpu}:${thin_gpu} --export=ALL,VALUE_MODEL=linear \
  scripts/slurm_thinning_benchmark.sh stack)
```

The outputs are under `artifacts/paper_evidence/thinning/`: the inputs in
`data/<dataset>/`, the teachers, fused value and selector of each unit in
`<unit>/`, the comparators in `comparators/<dataset>/`, and the evaluation in
`evaluation/matched_fraction_summary.{csv,json}`, which holds the weighted
kNN comparison for the thinning-trained selectors.

### 8.2 Thinning transfer

`scripts/slurm_thinning_transfer.sh` adds the Norman screen thinned to 50%
(from the benchmark of section 2.4) and a mask-trained design: the
fitting cells receive the stratified 10% mask on top of their thinned counts,
the teachers, fused value and selector are fitted as in the main method, and
the held-out thinning positives are evaluated. It also trains the selector on
standard scVI and standard MAGIC under both designs.

```bash
read -r tt_prep tt_cpu tt_gpu tt_sel tt_stacked tt_eval <<< \
  "$(bash scripts/slurm_thinning_transfer.sh submit | sed 's/[a-z]*=//g')"
```

Submit it after `thin_sel`, `thin_eval` and `lf_norman` have finished (the
`submit` stage sets dependencies only among its own stages). The stages are
`prepare` (array 0-5), `cpu` (0-34), `gpu` (0-13, L40S), `selector` (0-6),
`stacked` (0-23) and `evaluate`. The outputs are under
`artifacts/paper_evidence/review_round2/thinning_transfer/`:

- `evaluation/transfer_summary.{csv,json}` and
  `evaluation/transfer_paired_differences.csv`: the thinning comparisons
  other than weighted kNN, the count-one recall at 5%, the expected-count
  baseline and the share of positives with an original count of one
  (`composition.positive_count_1_share`).
- `mask_trained/<unit>/input/manifest.json`: the share of masked positives
  of the thinned fitting cells with a thinned count of one
  (`fitting_masked_count_1_share`).

### 8.3 Recall by count stratum

```bash
read -r cr_stacked cr_eval <<< \
  "$(bash scripts/slurm_count_stratified_recall.sh submit | sed 's/[a-z]*=//g')"
```

Submit it after `lf_evaluate` has finished. The `stacked` stage (array 0-9,
task = 2 × unit + method) trains the selector on standard scVI or standard
MAGIC for each unit of `leakage_free/units_manifest.json` and saves its
held-out scores (`stacked_selector_scores.py`). The `evaluate` stage runs
`evaluate_count_stratified_recall.py` and writes Supplementary Table S11 to
`artifacts/paper_evidence/review_round2/thinning_transfer/count_stratified_recall/count_stratified_recall.{csv,json}`.

## 9. Linear and autoencoder value comparisons

```bash
stack_lin=$(sbatch --parsable --array=0-3 --dependency=afterok:${teachers}:${scvi} \
  --export=ALL,VALUE_MODEL=linear scripts/slurm_safe_fusion_stack.sh)
autoenc=$(sbatch --parsable --dependency=afterok:${teachers}:${scvi} scripts/slurm_autoencoder_fusion.sh)
NV=scripts/slurm_norman_rebuilt_value.sh
nv_linear=$(sbatch --parsable --array=0 --dependency=afterok:${lf_stack} --export=ALL,STAGE=linear $NV)
nv_autoenc=$(sbatch --parsable --array=0-1 --dependency=afterok:${nv_linear} --export=ALL,STAGE=autoencoder $NV)
sbatch --array=0 --mem=64G \
  --dependency=afterok:${stack}:${stack_lin}:${autoenc}:${selectors}:${selector_rep}:${rep_lin}:${thin_sel}:${thin_lin}:${nr_selector}:${nr_lin}:${nv_autoenc}:${lf_selector} \
  --export=ALL,STAGE=evaluate $NV
```

- `stack_lin` writes the linear combination of the teachers
  (`<methods_root>/safe_fusion_linear`) for units 0-3 (the pancreas folds and
  colon).
- `slurm_autoencoder_fusion.sh` (array 0-7, task = 2 × unit + mode) fits the
  autoencoder fusion network of `fusion/` through
  `scripts/run_autoencoder_fusion.py` for the pancreas folds and colon. Mode
  `resampled` hides a fresh 15% of the recorded nonzero entries of the
  training cells in each epoch and recomputes the gene median, SVD and kNN
  proposals (`<methods_root>/autoencoder_fusion_3teachers`). Mode
  `masked_positives` learns from the masked positives with the five teacher
  contracts (`<methods_root>/autoencoder_fusion_5teachers`).
- `slurm_norman_rebuilt_value.sh` links the Norman teachers and fused value of
  section 3.4 into
  `artifacts/paper_evidence/review_round2/norman_rebuilt/methods/`, fits the
  linear combination (`linear`) and the two autoencoder networks
  (`autoencoder`) there, and runs `scripts/evaluate_value_accuracy.py`
  (`evaluate`) on the pancreas folds, colon and Norman with their mask
  replicates and the thinning units of section 8.1. It keeps the zeros
  selected by Safe Fusion fixed and inserts each candidate value.

The outputs are in
`artifacts/paper_evidence/review_round2/norman_rebuilt/value_accuracy/`:

- `error_removed.csv`: the error removed at 1% to 10% and the paired
  differences between values with bootstrap intervals, including the boosted
  fused value minus the linear combination at 5%.
- `log_error.csv`: the mean absolute log error.
- `log_error_strata.csv`: the error by true count and gene detection rate.
- `cross_validation.csv`: the out-of-fold error of the boosted and linear
  value models (Supplementary Section S6).
- `replicates.csv` and `summary.json`: the mask replicate and thinning units
  and the bootstrap summaries.

## 10. Deployment inputs and inductive downstream evaluation

### 10.1 Inductive evaluation on recorded counts of held-out cells

The deployment analysis applies every method to the recorded counts of the
held-out cells. The training cells keep the benchmark mask, so the selector
still learns from their masked positives. `scripts/deployment_paths.sh`
defines the eight units (`DEPLOY_KEYS`: the pancreas folds, colon, the three
screens and zebrafish) and `DEPLOY_ROOT` (default
`artifacts/paper_evidence/downstream_deployment`).

```bash
prepare=$(sbatch --parsable scripts/slurm_deployment_prepare.sh)
scvi_dep=$(sbatch --parsable --dependency=afterok:$prepare scripts/slurm_deployment_scvi.sh)
deploy_fill=$(sbatch --parsable --dependency=afterok:$scvi_dep scripts/slurm_deployment_fill.sh)
deploy_eval=$(sbatch --parsable --dependency=afterok:$deploy_fill scripts/slurm_deployment_evaluate.sh)
```

After `deploy_eval` finishes, run on a compute node:

```bash
python scripts/summarize_fill_evaluations.py \
  --root artifacts/paper_evidence/downstream_deployment/evaluation \
  --output artifacts/paper_evidence/downstream_deployment/summary.csv
```

- `slurm_deployment_prepare.sh` (array 0-7) runs `build_deployment_inputs.py`,
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
- `slurm_deployment_evaluate.sh` (array 0-7) evaluates them with the
  downstream evaluators of section 10.2, with the unfilled recorded matrix as
  the reference, and writes
  `artifacts/paper_evidence/downstream_deployment/evaluation/`.

`summary_key_metrics.csv` next to `summary.csv` holds the colon, pancreas and
zebrafish rows of the earlier inductive recorded-count analysis. The pancreas disease-effect row is the
`AAB_vs_Control` contrast of `disease_logfc_spearman`.

### 10.2 Inductive downstream evaluation on the masked benchmark

The downstream benchmark uses the donor-held-out pancreas and colon data, the
held-out zebrafish cells, and the Adamson, Dixit and Papalexi interventions.
Safe Fusion, SVD and weighted kNN fill 1% to 10% of the candidate zeros.
`MARKER_PANEL` (default `source`, the markers of the source studies in
`src/safefusion_benchmark/marker_panels.py`) selects the marker panel.
[COMPLETE_DOWNSTREAM_PROTOCOL.md](COMPLETE_DOWNSTREAM_PROTOCOL.md) defines the
tasks.

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

- `slurm_evaluate_complete_downstream.sh` (array 0-7) writes
  `artifacts/paper_evidence/downstream_complete/{clustering,markers,pancreas_biology,trajectory,grn}/`.
  `slurm_finalize_complete_downstream.sh` writes the combined tables
  (`all_bootstrap_summaries.csv`, the inductive downstream results except
  Norman) and a five-panel figure to `artifacts/paper_evidence/downstream_complete/summary/`.
  It fails unless all five tasks, all ten fill fractions, all 24 pancreas
  donors, all 9 colon donors, all 12 zebrafish stages and the 64 regulators of
  the three screens are present and pass the leakage checks.
- `slurm_decompose_fills.sh` (array 0-8, task = 3 × dataset + source, with
  datasets colon, pancreas and zebrafish and sources Safe Fusion, SVD and
  weighted kNN) splits each fill into its selected masked positives and its
  selected recorded zeros (`decompose_fills.py`, which writes
  `decomposition_counts.csv`) and evaluates each part alone. The parts and
  their evaluations are in
  `artifacts/paper_evidence/downstream_decomposition/<dataset>/<source>/`, and
  `summary.csv` and `summary_key_metrics.csv` collect the comparisons with the
  masked input. The Results text on masked positives and recorded zeros comes
  from these files. The pancreas share of masked positives pools the
  `decomposition_counts.csv` of the three folds.

### 10.3 Inductive fill-fraction analysis for section 30

`slurm_detection_rule.sh` refits the selector with the production settings,
calibrates its scores by isotonic regression on cross-fitted scores of the
training cells, and fills every held-out zero whose detection probability
p / (p + ρ(1 − p)) exceeds 1/2, with ρ = 0.10
(`--detection-rule-mask-rate 0.10`). Tasks 0-3 (colon, the pancreas folds)
apply the rule to the masked benchmark. Tasks 4-7 apply it to the deployment
input of section 10.1, where every held-out candidate is a recorded zero.

```bash
sbatch --array=0-3 --dependency=afterok:${stack} scripts/slurm_detection_rule.sh
sbatch --array=4-7 --dependency=afterok:${deploy_fill} scripts/slurm_detection_rule.sh
```

The `detection_rule` block of
`artifacts/paper_evidence/detection_rule/<unit>/calibration_report.json`
holds the chosen fill fraction, its masked F1, and the fraction and F1 with
the highest F1 in hindsight. In
`artifacts/paper_evidence/downstream_deployment/{colon,pancreas/fold_<k>}/detection_rule/calibration_report.json`,
`detection_rule.test_fill_fraction` is the share of recorded zeros that the
rule fills (last column of Table S26).

### 10.4 The Norman screen

`scripts/slurm_norman_rebuilt_downstream.sh` repeats the Norman tasks of the
launchers of sections 10.1 to 10.3 on the benchmark of section 2.4, with the
selector of section 3.4. `STAGE` selects the step.

```bash
ND=scripts/slurm_norman_rebuilt_downstream.sh
nd_matched=$(sbatch --parsable --mem=32G --dependency=afterok:${lf_selector} --export=ALL,STAGE=matched $ND)
nd_masked=$(sbatch --parsable --time=04:00:00 --dependency=afterok:${nd_matched} --export=ALL,STAGE=masked_grn $ND)
nd_rule=$(sbatch --parsable --mem=24G --dependency=afterok:${lf_stack} --export=ALL,STAGE=detection_masked $ND)
nd_prep=$(sbatch --parsable --dependency=afterok:${lf_norman} --export=ALL,STAGE=deploy_prepare $ND)
nd_scvi=$(sbatch --parsable --mem=32G --gres=gpu:l40s:1 --dependency=afterok:${nd_prep} --export=ALL,STAGE=deploy_scvi $ND)
nd_fill=$(sbatch --parsable --dependency=afterok:${nd_scvi} --export=ALL,STAGE=deploy_fill $ND)
nd_eval=$(sbatch --parsable --time=04:00:00 --dependency=afterok:${nd_fill} --export=ALL,STAGE=deploy_eval $ND)
nd_rule_dep=$(sbatch --parsable --mem=24G --dependency=afterok:${nd_fill} --export=ALL,STAGE=detection_deploy $ND)
sbatch --dependency=afterok:${nd_masked}:${nd_rule}:${nd_eval}:${nd_rule_dep} --export=ALL,STAGE=summarize $ND
```

The outputs are under
`artifacts/paper_evidence/review_round2/norman_rebuilt/`: the SVD and weighted
kNN fills in `matched_fraction/`, the response-edge evaluation on the masked
benchmark in `downstream_masked/grn/norman_crispra/`, the deployment input,
fills and evaluation in `deployment/`, and the rule in
`detection_rule/norman_crispra/` and
`deployment/norman_crispra/detection_rule/`. The `summarize` stage runs
`summarize_norman_rebuilt_downstream.py` and writes
`norman_downstream_rows.csv`, whose stored `table` keys are `3`, `S17` and `S21`. The `S21` key supplies
the Norman row of final Table S26; the other keys supply inductive reference endpoints.

## 11. Inductive inserted-value evaluation

```bash
read -r iv_prep iv_scvi iv_fill iv_eval <<< \
  "$(bash scripts/slurm_inserted_value.sh submit ${tt_eval}:${deploy_fill} | sed 's/[a-z]*=//g')"
```

Submit it after `lf_norman` has finished. The optional argument of `submit` is
a colon-separated list of job IDs that the `evaluate` stage waits for. The
`prepare`, `scvi` and `fill` stages build the Norman deployment input
from the benchmark of section 2.4 under
`artifacts/paper_evidence/review_round2/inserted_value/deployment/norman_crispra/`,
with the commands of section 10.1. The `evaluate` stage runs
`evaluate_inserted_value.py all` and writes, in
`artifacts/paper_evidence/review_round2/inserted_value/`:

- `recorded_zero_fills.csv`: the median value that Safe Fusion and SVD insert
  into recorded zeros of held-out cells and the share of filled zeros whose
  expected count exceeds 2. It reads the production deployment of
  section 10.1 for every dataset except Norman.
- `thinning_positive_bias.csv`: the signed bias against the count expected at
  the thinned depth (rows with `design` mask-trained and
  `recorded_count` all) and the absolute log error against the count before
  thinning (`abs_error_vs_recorded_count`, Supplementary Section S6), from
  the thinning units of sections 8.1 and 8.2.

## 12. Disease, sex and annotation checks

These checks use the recorded counts of the held-out tissue cells from
section 10.1 and the masked tissue benchmark.

```bash
dc_fill=$(sbatch --parsable --dependency=afterok:${deploy_fill}:${stack} scripts/slurm_disease_control_fill.sh)
DC=scripts/slurm_disease_control_analysis.sh
sbatch --array=2-3 --dependency=afterok:${dc_fill} $DC
sbatch --array=0,1,4-9 --cpus-per-task=4 --mem=32G --time=01:00:00 \
  --dependency=afterok:${dc_fill}:${curves}:${baselines} $DC
```

- `slurm_disease_control_fill.sh` (array 0-3: the pancreas folds, colon) fills
  the recorded zeros of the held-out cells with the inductive MAGIC and scVI
  teachers at 1%, 5% and 10% (`deployment_fills/`), and refits the Safe Fusion
  selectors of the deployment analysis and of the masked benchmark with their
  production arguments and seed to write the scores of every XIST and RPS4Y1
  zero (`selector_scores/{deployment,masked}/`). The refitted selectors
  reproduce the production fills.
- `slurm_disease_control_analysis.sh` (array 0-9, task = 2 × analysis +
  tissue, with analyses condition, effects, modules, annotation and
  sex_zeros and tissues pancreas and colon) runs
  `scripts/disease_control_<analysis>.py`. The modules and annotation tasks
  use `.venv-scanpy`, which the launcher creates when it is missing.

The outputs are under `artifacts/paper_evidence/disease_control_checks/`:

- Sex-linked zero controls:
  `sex_zeros/<tissue>/auroc.csv`, with `fill_rates.csv`, `zero_counts.csv`
  and `donor_sex.csv` for the donor-sex paragraph.
- Disease effects: `disease_effects/<tissue>/observed_summary.csv` (slopes and
  discoveries) and `permutation_null.csv` (false discoveries under permuted
  donor labels).
- Module scores: `module_scores/<tissue>/disease_effect.csv`.
- Reference mapping: `annotation/<tissue>/reference_mapping_overall.csv`.
- `condition/<tissue>/depth.csv` and `fill_rate.csv`: library size, zero
  fraction and fill rate per condition (Supplementary Section S10).

## 13. Perturbation analyses and label controls

This analysis ranks the zeros of the perturbed gene in the recorded counts of
the held-out screen cells and reports masked F1 on the four masked screen
benchmarks. Adamson, Dixit and Papalexi use the production benchmarks and the
deployment inputs of section 10.1. Norman uses the benchmark of section 2.4.

### 13.1 Perturbation labels on Adamson, Dixit and Papalexi

`slurm_condition_aware_screens.sh` fits Safe Fusion with perturbation labels
(obs column `target`). Array task = 2 × screen + setting, with screens
Adamson, Dixit and Papalexi. Even tasks use the masked benchmark, and odd
tasks use the deployment input. `STAGE=knn` fits the weighted kNN teacher
within each label, `STAGE=scvi` the scVI teacher with the label as a
covariate, and `STAGE=select` the fused value from the gene median, SVD, MAGIC
and the two label teachers, followed by the selector with the two label
features. `STAGE=matched` fills the deployment zeros at 1% to 10% with the
label kNN and scVI teachers and with the MAGIC and scVI teachers, each ranked
by its own value.

```bash
C=scripts/slurm_condition_aware_screens.sh
knn_c=$(sbatch --parsable --array=0-5 --dependency=afterok:${prepare} --export=ALL,STAGE=knn $C)
scvi_c=$(sbatch --parsable --array=0-5 --gres=gpu:l40s:1 --dependency=afterok:${prepare} --export=ALL,STAGE=scvi $C)
select_c=$(sbatch --parsable --array=0-5 --dependency=afterok:${teachers}:${prepare}:${knn_c}:${scvi_c} --export=ALL,STAGE=select $C)
matched_c=$(sbatch --parsable --array=1,3,5 --dependency=afterok:${prepare}:${scvi_dep}:${knn_c}:${scvi_c} --export=ALL,STAGE=matched $C)
```

The masked setting writes `graph_smooth_condition`,
`scvi_inductive_condition`, `safe_fusion_condition` and `selector_condition`
under `<methods_root>`. The deployment setting writes the same teachers to
`methods/` and the fills to `selector_condition/` and `matched_condition/` in
`artifacts/paper_evidence/downstream_deployment/<screen>/`.

### 13.2 Norman, standard imputers, selector scores, null test and Mixscape

```bash
K=scripts/slurm_knockdown_norman_rebuilt.sh
kn_knn=$(sbatch --parsable --dependency=afterok:${lf_stack} --export=ALL,STAGE=masked_knn $K)
kn_scvi=$(sbatch --parsable --gres=gpu:l40s:1 --dependency=afterok:${lf_stack} --export=ALL,STAGE=masked_scvi $K)
kn_select=$(sbatch --parsable --dependency=afterok:${kn_knn}:${kn_scvi} --export=ALL,STAGE=masked_select $K)
kn_prep=$(sbatch --parsable --dependency=afterok:${lf_norman} --export=ALL,STAGE=deploy_prepare $K)
kn_dscvi=$(sbatch --parsable --gres=gpu:l40s:1 --dependency=afterok:${kn_prep} --export=ALL,STAGE=deploy_scvi $K)
kn_fill=$(sbatch --parsable --dependency=afterok:${kn_prep}:${kn_dscvi} --export=ALL,STAGE=deploy_fill $K)

S=scripts/slurm_knockdown_standard_imputers.sh
after_std=afterok:${kn_prep}:${prepare}
kd_std=$(sbatch --parsable --dependency=${after_std} --export=ALL,METHOD=alra $S)
kd_std=${kd_std}:$(sbatch --parsable --dependency=${after_std} --export=ALL,METHOD=magic $S)
kd_std=${kd_std}:$(sbatch --parsable --dependency=${after_std} --export=ALL,METHOD=saver $S)
kd_std=${kd_std}:$(sbatch --parsable --gres=gpu:l40s:1 --dependency=${after_std} --export=ALL,METHOD=scvi $S)

kd_scores=$(sbatch --parsable --dependency=afterok:${deploy_fill}:${select_c} scripts/slurm_knockdown_selector_scores.sh)

N=scripts/slurm_knockdown_pseudolabel_null.sh
p=$(sbatch --parsable --dependency=afterok:${deploy_fill}:${select_c} --export=ALL,STAGE=prepare $N)
k=$(sbatch --parsable --dependency=afterok:$p --export=ALL,STAGE=knn $N)
v=$(sbatch --parsable --dependency=afterok:$p --gres=gpu:l40s:1 --export=ALL,STAGE=scvi $N)
s=$(sbatch --parsable --dependency=afterok:$k:$v --export=ALL,STAGE=select $N)
kd_null=$(sbatch --parsable --array=0 --dependency=afterok:$s --export=ALL,STAGE=evaluate $N)

mixscape=$(sbatch --parsable --dependency=afterok:${fetch} scripts/slurm_papalexi_mixscape.sh)

sbatch --dependency=afterok:${kn_select}:${kn_fill}:${kd_std}:${kd_scores}:${mixscape}:${select_c}:${matched_c}:${deploy_fill}:${selectors}:${lf_selector} \
  scripts/slurm_evaluate_knockdown_zero_analyses.sh
```

- `scripts/knockdown_paths.sh` sets the paths of each screen. For Norman, the
  label teachers of the masked benchmark go to
  `artifacts/paper_evidence/review_round2/knockdown/norman_rebuilt/masked/`
  and the deployment input, teachers and fills to
  `knockdown/norman_rebuilt/deployment/`. The deployment selectors of
  `slurm_knockdown_norman_rebuilt.sh` also write the score of every zero of the
  target genes (`--score-gene`).
- `slurm_knockdown_standard_imputers.sh` (array 0-2: Norman, Adamson,
  Papalexi) fits ALRA, standard MAGIC, SAVER and standard scVI on each
  deployment input with the settings of section 3.3.
- `slurm_knockdown_selector_scores.sh` (array 0-3, task = 2 × screen +
  variant, with screens Adamson and Papalexi and variants without and with
  labels) refits the deployment selectors and writes the score of every zero
  of the target genes.
- `slurm_knockdown_pseudolabel_null.sh` (array 0-3, the pseudo-label
  assignments with seeds 1729 to 1732) gives half of the Adamson control cells
  random pseudo-labels, refits the label-aware Safe Fusion, and tests
  differential expression between groups (`knockdown_pseudolabel_null.py`).
  Its evaluation is in
  `artifacts/paper_evidence/review_round2/knockdown/pseudolabel_null/adamson_crispri/evaluation/`
  (`summary.json`, `pseudo_group_de.csv`).
- `slurm_papalexi_mixscape.sh` runs Mixscape on the Papalexi MuData object in
  `.venv-pertpy` (`run_papalexi_mixscape.py`) and writes
  `knockdown/mixscape/papalexi_mixscape_classes.parquet`.
- `slurm_evaluate_knockdown_zero_analyses.sh` runs
  `evaluate_condition_masked_f1.py` and
  `evaluate_knockdown_zero_analyses.py --depth-strata 5`.

The outputs are under `artifacts/paper_evidence/review_round2/knockdown/`:

- `evaluation/report.json`: in `screens.<screen>.auroc["<method>|fill_order"]`,
  the unadjusted AUROC (`none`) and the AUROC within library-size quintiles
  (`depth_strata`) of the earlier inductive zero-control analysis, and in
  `knockdown_effect_shift["<method>|10"]` the shift of the held-out
  perturbation log2 fold change at 10%. It also holds the
  library-size AUROC, the continuous AUROC, the comparison of the Adamson and
  Papalexi screens, the held-out fold changes (`heldout_log2fc_recorded`) and
  the Mixscape fill rates (`mixscape_papalexi`) of Supplementary Section S7.
  `auroc.csv`, `effects.csv`, `fills.csv`, `targets.csv`, `mixscape.csv` and
  `per_target_table.csv` hold the per-target values. The fill rates of all
  Papalexi control zeros quoted next to the Mixscape classes pool the
  `fill_control` values of `fills.csv` (method `safe_fusion`) over targets,
  weighted by `n_control_zero` of `targets.csv`.
- `masked_f1/masked_f1_report.json`: the mean masked F1 over the fill
  fractions 1% to 10%, averaged over the four screens in the earlier
  inductive analysis, and `masked_f1/masked_f1_by_screen.json` the value of each screen
  with and without labels, with intervals over perturbation labels.

Section 12 provides the earlier inductive sex-linked zero controls.

## 14. Earlier inductive RNA-protein analysis

This section and `data/papalexi_crossmodal_inputs.tar.gz` reproduce the earlier
inductive protein analysis (Safe Fusion pooled Spearman 0.530). They do not
reproduce the manuscript's Figure 2 or Supplementary Tables S19 and S20.
Use the saved-input quick path in section 22 for those results.

```bash
after=afterok:${audit}:${pdl1_scvi}:${pdl1_selector}
sbatch --dependency=${after} scripts/slurm_evaluate_papalexi_cd274.sh
sbatch --dependency=${after} scripts/slurm_evaluate_papalexi_crossmodal.sh
sbatch --dependency=${after} scripts/slurm_evaluate_papalexi_crossmodal_raw.sh
sbatch --dependency=${after} scripts/slurm_evaluate_protein_within_state.sh
```

- `slurm_evaluate_papalexi_cd274.sh` runs `evaluate_papalexi_crossmodal.py`
  for CD274 on centered log ratio and raw PD-L1
  (`benchmark/evaluation_cd274/`, `benchmark/evaluation_cd274_raw_counts/`),
  then `evaluate_pdl1_state_baselines.py` (`benchmark/evaluation_pdl1_state/`).
  It then plots the earlier inductive protein analysis with
  `plot_biological_range_figures.py --pdl1-only` in `.venv-baselines`, which writes `pdl1_range_validation.{png,pdf}` to
  `artifacts/paper_evidence/figures/`. `PAPER_DIR=<dir>` writes an extra PNG
  copy to `<dir>`.
- `slurm_evaluate_papalexi_crossmodal.sh` and
  `slurm_evaluate_papalexi_crossmodal_raw.sh` evaluate CD86, PD-L2 and TIM-3
  together with CD274, on centered log ratio and raw counts
  (`benchmark/evaluation/`, `benchmark/evaluation_raw_counts/`).
- `slurm_evaluate_protein_within_state.sh` refits the Papalexi selector with
  global fill fractions of 1%, 5% and 10% and scores every gene
  (`review_round2/protein/global_fill_selector/`), then runs
  `evaluate_protein_within_state.py`, which writes
  `artifacts/paper_evidence/review_round2/protein/evaluation/`.

The `benchmark/` paths are under
`artifacts/paper_evidence/papalexi_crossmodal/`. In each evaluation
directory, `protein_threshold_range_metrics.csv` holds the threshold AUROC,
`continuous_protein_association.csv` the Spearman correlations,
`fill_range_protein_enrichment.csv` and `range_leaders.csv` the enrichment of
the filled cells and the methods with the highest value at each threshold and
fraction, `replicate_association.csv` the correlation within each replicate,
and `permutation_tests.csv` the permutation tests. For this earlier inductive
analysis, the replicate correlations come from
`evaluation_cd274/replicate_association.csv`, the raw column from `evaluation_cd274_raw_counts/continuous_protein_association.csv`,
and the CD86, PD-L2 and TIM-3 columns from
`evaluation/continuous_protein_association.csv`. In
`review_round2/protein/evaluation/`, `pdl1_pooled_rankings.csv` holds the
pooled correlations, mean AUROC and paired differences with intervals over
perturbation targets. `association.csv` holds the correlations within targets,
in the control cells, after adjustment (`partial_spearman`) and of detected
RNA with protein. `global_fill.csv` holds the
CD274 zeros filled at global fill fractions.

### Earlier inductive protein analysis from saved inputs

`data/papalexi_crossmodal_inputs.tar.gz`, stored with Git LFS, holds the
inputs of these evaluations: the prepared RNA-protein file, the audit panel,
the masked benchmark, the five teacher contracts, the fused value, standard
scVI and the selector scores (see [data/README.md](../data/README.md) for
the checksum and contents). Unpack it at the repository root and run the
evaluations below. The within-state launcher refits the selector:

```bash
git lfs pull --include "data/papalexi_crossmodal_inputs.tar.gz"
tar -xzf data/papalexi_crossmodal_inputs.tar.gz
sbatch scripts/slurm_evaluate_papalexi_cd274.sh
sbatch scripts/slurm_evaluate_papalexi_crossmodal.sh
sbatch scripts/slurm_evaluate_papalexi_crossmodal_raw.sh
sbatch scripts/slurm_evaluate_protein_within_state.sh
```

These launchers need `.venv`, and plotting also needs `.venv-baselines`
(section 1). The within-state launcher refits the inductive selector on an
Intel Xeon Platinum 8592+ node before evaluation.

## 15. Standard-imputer fill rates

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

## 16. Benchmark counts for shared inputs

```bash
sbatch --dependency=afterok:${pdl1_prep} scripts/slurm_summarize_benchmark_data.sh
sbatch --array=0 --dependency=afterok:${lf_norman}:${lf_pancreas} --export=ALL,STAGE=benchmark_data $LF
```

Both launchers count the cells, genes, held-out candidates and masked
positives of their units with `summarize_benchmark_data.py`. They need the
inputs of section 2 only.
`slurm_summarize_benchmark_data.sh` covers the production units and writes
`artifacts/paper_evidence/benchmark_data/benchmark_data.csv`, which holds the
colon, screen, zebrafish and RNA-protein rows of Table S2 and, in its
pancreas rows, the candidate counts of the production gene set given in the
caption. The `benchmark_data` stage of the leakage-free launcher covers the
pancreas folds and the Norman screen and writes
`leakage_free/benchmark_data/benchmark_data.csv`.

## 17. Comparison-method settings

Table S3 lists the settings of the comparison launchers of sections 3.3 and
3.4. The ALRA rank is `chosen_k` in the `metadata.json` of each ALRA contract:
`artifacts/paper_evidence/baselines/alra/colon_mask_010` for colon and
`leakage_free/baselines/alra/{pancreas_fold_<k>,norman}` for the pancreas
folds and the Norman screen.

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
25 epochs. `sbatch_scgcl_baselines.s` fits pancreas and colon, resumes from
the checkpoint after preemption and runs `scripts/evaluate_scgcl_baseline.py`
on each fit. For the Norman screen:

```bash
scgcl_norman=$(sbatch --parsable --cpus-per-task=16 --mem=96G --dependency=afterok:${lf_norman} \
  --export=ALL,STAGE=scgcl $NS)
sbatch --dependency=afterok:${scgcl_norman}:${lf_evaluate} --export=ALL,STAGE=scgcl_compare $NS
```

The `scgcl` stage fits scGCL on the Norman benchmark
(`review_round2/norman_rebuilt/baselines/scgcl/norman`, 2,048 variable genes),
and `scgcl_compare` computes its masked F1 curve and the paired bootstrap in
`review_round2/norman_rebuilt/scgcl_comparison/`. The differences between Safe
Fusion and scGCL averaged over the fill fractions 1% to 10% are the scGCL rows
with statistic `mean_difference_1_to_10` of
`artifacts/paper_evidence/masked_f1_paired_bootstrap.csv` (pancreas and colon,
section 4) and of `scgcl_comparison/masked_f1_paired_bootstrap.csv` (CRISPRa).
Two further numbers of Section S3 are computed from these outputs. The ratio of
scGCL to a random ranking divides the mean scGCL F1 over the fill fractions
1% to 10% (`selector_f1_fillrate_baselines_1000_points.csv` and
`scgcl_comparison/scgcl_1000_points.csv`) by the mean F1 of a random ranking,
2πb / (π + b) at fill fraction b and prevalence π (Table S2). The decrease of
the training loss compares the first and the last loss recorded in the
`metadata.json` of each scGCL contract.

## 18. Runtime

```bash
python3 scripts/summarize_runtime.py
```

`scripts/summarize_runtime.py` reads `scripts/runtime_jobs.tsv`, which lists
the Slurm tasks of the paper run that fitted each step for each dataset
(sections 3.2 and 3.4), and queries `sacct` for their elapsed time and peak
memory. To summarize your own run, replace the job IDs in
`scripts/runtime_jobs.tsv` with those of your `teachers`, `scvi`, `stack` and
`selectors` jobs and, for Norman, of `lf_teachers`, `lf_scvi_teacher`,
`lf_stack` and `lf_selector`. `--sacct-file` reads a saved pipe-delimited
`sacct` dump instead of querying Slurm. The script writes `runtime_table.csv`
(Table S27), `runtime_by_dataset.csv` and `sacct_records.txt` to
`artifacts/paper_evidence/runtime/`.


## 19. Final donor folds and comparison methods

Prerequisites: sections 1–18, including rebuilt pancreas and Norman inputs,
inductive deployment inputs, perturbation controls and RNA-protein fits.
Colon uses three folds over all 34 donors. Pancreas uses the rebuilt fold-specific
gene panels. These commands also build both modes' teachers and selectors.

```bash
export SBATCH_ACCOUNT=torch_pr_634_general
mkdir -p logs
bash scripts/slurm_colon_crossfit.sh submit
```

After those jobs finish:

```bash
bash scripts/slurm_colon_crossfit_thinning.sh submit
bash scripts/slurm_pancreas_rebuilt_thinning.sh submit
sbatch --wait -A torch_pr_634_general scripts/slurm_r3_comparators_setup.sh
sbatch --wait --array=0-15 --export=ALL,STAGE=kcluster scripts/slurm_r3_comparators.sh
for method in dca dca_zinb scimpute screcover enimpute; do
  sbatch --wait --array=0-15 --export=ALL,STAGE=fit,METHOD=$method scripts/slurm_r3_comparators.sh
done
sbatch --wait --array=0-15 -p l40s_public --gres=gpu:l40s:1 --export=ALL,STAGE=scvi_zinb scripts/slurm_r3_comparators.sh
sbatch --wait --export=ALL,STAGE=units scripts/slurm_r3_comparators_evaluate.sh
sbatch --wait --array=0-47 --export=ALL,STAGE=stacked scripts/slurm_r3_comparators_evaluate.sh
for stage in fills masked masked_cf known_zeros; do
  if [[ $stage == fills ]]; then array=0-5; else array=0; fi
  sbatch --wait --array=$array --export=ALL,STAGE=$stage scripts/slurm_r3_comparators_evaluate.sh
done
```

DCA, scImpute, scRecover, EnImpute and zero-inflated variants use
`scripts/standard_imputers/run_r3_comparator.py`, its R/Python runners and
`scripts/standard_imputers/r3_units.sh`. Table S3 is a settings table, read from
these launchers and the comparison launchers in section 17. It is not fitted
or generated by a table-writing script.

Table S2 reads the production and leakage-free `benchmark_data.csv` files of
section 16, with the colon rows replaced by these fold counts:

```bash
args=()
for fold in 0 1 2; do
  d=artifacts/paper_evidence/review_round3/colon_crossfit/fold_$fold
  args+=(--unit colon_$fold "$d/corrupted.h5ad" "$d/coordinates.parquet" "$d/splits.parquet")
done
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/summarize_benchmark_data.py "${args[@]}" --output artifacts/paper_evidence/review_round3/colon_crossfit/benchmark_data.csv
```

The comparator evaluations write
`artifacts/paper_evidence/review_round3/comparators/evaluation/{masked,masked_colon_crossfit}/{absolute,paired_differences}.csv`.
These feed the final masked comparisons in section 21. Package versions are pinned in the setup launchers, which install the required environments.

## 20. Inductive settings, training targets and alternative values

Prerequisites: section 19 and its completed thinning jobs. These analyses supply
Table S8, the molecule-masking column of Table S9 and the value references used
by section 21. The selection rule uses inner validation; test outputs do not
select a model.

```bash
S=scripts/slurm_v2_selector.sh
sbatch --wait --array=0-39 -c 2 --mem=24G --time=00:30:00 "$S" inner_inputs
sbatch --wait --array=0-159 --mem=32G "$S" inner_cpu
sbatch --wait --array=0-39 -p l40s_public --gres=gpu:l40s:1 --mem=32G "$S" inner_gpu
sbatch --wait --array=0-79 -p cs "$S" inner_selector
sbatch --wait -c 2 --mem=8G --time=00:20:00 "$S" inner_evaluate
T=scripts/slurm_v2_selector_test.sh
sbatch --wait --array=0-27 -c 2 --mem=24G --time=00:30:00 "$T" inputs
sbatch --wait --array=0-111 --mem=32G "$T" cpu
sbatch --wait --array=0-27 -p l40s_public --gres=gpu:l40s:1 --mem=32G "$T" gpu
sbatch --wait --array=0-27 -p cs "$T" selector
sbatch --wait --array=0-7 -p cs "$T" current_selector
sbatch --wait --array=0-23 --mem=32G "$T" trans_cpu
sbatch --wait --array=0-7 --mem=32G "$T" trans_magic
sbatch --wait --array=0-7 -p l40s_public --gres=gpu:l40s:1 --mem=32G "$T" trans_scvi
sbatch --wait --array=0-7 -p cs "$T" trans_selector
sbatch --wait -c 8 --mem=64G --time=03:00:00 "$T" evaluate
```

The target choice is
`artifacts/paper_evidence/review_round3/selector_v2/inner/evaluation_colon_crossfit/selection.json`.
Table S9 reads `test/evaluation/masked/{absolute,paired_differences}.csv`,
`test/evaluation/thinned/{summary,paired_differences}.csv`
and the known-zero and protein tables written by `v2_selector_test_evaluate.py`
under `test/evaluation/references/`: `knockdown_summary.csv`,
`sex_paired.csv` and `pdl1_paired.csv`.

```bash
S=scripts/slurm_v2_ablation.sh
for stage in mask_input mask_teachers mask_scvi mask_stack mask_selector nocf_teachers nocf_scvi nocf_stack nocf_selector selector_size value_boosting; do
  gpu=()
  case $stage in
    mask_input|mask_stack|mask_selector) array=0-15 ;;
    mask_teachers) array=0-63 ;;
    mask_scvi) array=0-15; gpu=(-p l40s_public --gres=gpu:l40s:1) ;;
    nocf_teachers|selector_size) array=0-23 ;;
    nocf_scvi) array=0-7; gpu=(-p l40s_public --gres=gpu:l40s:1) ;;
    nocf_stack|nocf_selector) array=0-7 ;;
    value_boosting) array=0-47 ;;
  esac
  sbatch --wait -p cs "${gpu[@]}" --array=$array --export=ALL,STAGE=$stage "$S"
done
sbatch --wait -p cs --array=0 --cpus-per-task=4 --mem=64G --time=01:00:00 --export=ALL,STAGE=evaluate "$S"
V=scripts/slurm_v2_value.sh
sbatch --wait --array=0-15 --export=ALL,STAGE=inner "$V"
sbatch --wait --export=ALL,STAGE=select "$V"
sbatch --wait --array=0-10,20-28 --export=ALL,STAGE=fit "$V"
sbatch --wait --export=ALL,STAGE=evaluate "$V"
```

Table S8 reads
`artifacts/paper_evidence/review_round3/value_v2_ablations/ablations/evaluation/paired_differences.csv`.
The value selection is `value/selection/selection.json`; its held-out value
reference is `value/test/value_accuracy.csv`. `scripts/v2_value_models.py`
implements the conditional, detection-weighted and count-rate alternatives.

## 21. Main masked recovery, thinning and inserted values

Prerequisites: sections 19–20, including all comparison fits, the reference
fusion evaluation in section 5, and the Norman replicate fits in section 6.
The following stages use unchanged biological-unit bootstrap kernels.

```bash
O=artifacts/paper_evidence/review_round4/transductive_main
C=scripts/analyses/transductive_main
mkdir -p "$O/logs"
python "$C/submit.py" replicate
python "$C/submit.py" recorded
bash scripts/slurm_thinning_transductive.sh submit
```

Wait for all three chains. Then run:

```bash
sbatch --wait --array=0-48 "$C/run.sh" fit_s5.py
sbatch --wait --array=0-13 "$C/run.sh" attribution_exact.py
sbatch --wait --array=0-2 "$C/run.sh" masked.py 0
sbatch --wait "$C/run.sh" feature_eval.py
sbatch --wait "$C/run.sh" replicate_eval.py
sbatch --wait "$C/run.sh" thinning.py
sbatch --wait "$C/run.sh" value.py masked
sbatch --wait "$C/run.sh" value.py thinning
sbatch --wait "$C/run.sh" recorded_eval.py
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/build_transductive_fillrate_summary.py --output-dir "$O"
```

Files below are relative to `$O`:

| Item | Output read by the manuscript | Calculation |
|---|---|---|
| Table 1 | `masked/{Pancreas,Colon,CRISPRa}/{paired_differences,new_comparators_paired_differences}.csv`; `selector_f1_fillrate_transductive_1000_points.csv` | `masked.py`, `fusion_value_bootstrap.py`, `build_transductive_fillrate_summary.py` |
| Figure 1 | See section 29 | Final all-cell curves and plot |
| S4 | `rebuilt_replicates/evaluation/{per_seed,across_seeds,paired_differences,wins}.csv` | `replicate_eval.py` |
| S5 | `masked/{Pancreas,Colon,CRISPRa}/{paired_differences,new_comparators_paired_differences}.csv` | Both comparison tables, all three statistics |
| S6 | `item3_evaluation/{Pancreas,Colon,CRISPRa}/paired_differences.csv` | Transductive teacher-removal and classifier variants |
| S7 | `item3_evaluation/feature_pr_auc_exact.csv` | `attribution_exact.py`, `feature_eval.py` |
| S10 | `thinning_evaluation/{absolute,paired_differences}.csv` | `thinning.py`, thinning-trained and mask-trained designs |
| S11 | `artifacts/paper_evidence/review_round2/thinning_transfer/count_stratified_recall/count_stratified_recall.csv` | Inductive masked recall; section 8.3 |

The transductive value results in the surrounding text read
`value_accuracy/{masked,recorded,thinning}.csv`. S11 retains its inductive benchmark. S12–S15 now read the current outputs
in section 30, with historical autoencoder comparisons retained from section 9.

The transductive mode is the reference for the main comparisons. The inductive
rows retain their own scores, values, training splits and intervals. Counts of
winning fractions are descriptive counts, not bootstrap estimates.

## 22. Perturbation, sex and protein controls

### Quick path: Figure 2 and Supplementary Tables S19 and S20 from saved inputs

This path uses `data/papalexi_protein_current_inputs.tar.gz`. It needs only
Git LFS and the `.venv` and `.venv-baselines` environments from section 1.
It does not need the earlier protein archive or any teacher, fused-value or
selector refits. Run from the repository root, with unpacking on a compute
node. Set your Slurm account and partition as described under Conventions.

```bash
set -euo pipefail
git lfs install
git lfs pull --include "data/papalexi_protein_current_inputs.tar.gz"
echo "ff1082d83c9f67a578fa0304d5cc6838ce9df099cdedc66c38afb72db53be932  data/papalexi_protein_current_inputs.tar.gz" | sha256sum --check
tar -xzf data/papalexi_protein_current_inputs.tar.gz
mkdir -p logs
sbatch --wait --cpus-per-task=8 --mem=48G --time=06:00:00 \
  --export=ALL,STAGE=evaluate scripts/analyses/transductive_references/protein_cs.sh && \
sbatch --wait --cpus-per-task=8 --mem=48G --time=01:00:00 \
  --export=ALL,STAGE=plot scripts/analyses/transductive_references/protein_cs.sh
```

Compare the archive checksum with [data/README.md](../data/README.md) before
unpacking. The archive contains regular files at the paths read by the
launchers, including resolved copies of linked inputs. No setup script is
needed. The `evaluate` stage runs the protein, state-baseline and matched
within-state evaluators. The `plot` stage reads the new evaluation CSVs.
Both stages write under
`artifacts/paper_evidence/review_round4/transductive_references/protein_cs/`:

- Figure 2: `figures/pdl1_range_validation.png` and `.pdf`, from
  `evaluation_cd274/protein_threshold_range_metrics.csv` and
  `evaluation_cd274/fill_range_protein_enrichment.csv`.
- Supplementary Table S19: `evaluation_cd274/replicate_association.csv`,
  `evaluation_cd274_raw_counts/continuous_protein_association.csv` and
  `evaluation/continuous_protein_association.csv`.
- Supplementary Table S20: `evaluation_within_state/association.csv`,
  `pdl1_pooled_rankings.csv`, `paired_differences.csv` and `global_fill.csv`.

The expected Safe Fusion pooled Spearman correlation is 0.547 and the mean
AUROC across the 41 PD-L1 thresholds is 76.5%. The manuscript reports the
paired Safe Fusion minus SVD correlation as 0.040 [0.006, 0.080]. The saved
inputs give 0.039463 [0.006076, 0.079894], which rounds to
0.039 [0.006, 0.080]. The point estimate therefore differs from the printed
manuscript by 0.001; the interval agrees. The archive preserves the saved
values without adjustment.

### Full perturbation, sex and protein workflow

Prerequisites: sections 13–14 and 19–21. The setup script rebuilds input and
comparator links inside this clone. It does not copy data or contracts into Git.

```bash
bash scripts/analyses/transductive_references/reproduce.sh
```

This launcher waits at each stage: inductive evaluator checks; masked and
recorded perturbation fits; rebuilt tissue inputs; label-free and donor-aware
teachers/selectors; matched sex evaluation; then protein fitting, evaluation
and plotting. CPU selectors use `cs`; scVI uses L40S. The evaluators are
`evaluate_knockdown_zero_analyses.py`, `evaluate_condition_masked_f1.py`,
`sex_evaluate.py`, `evaluate_papalexi_crossmodal.py` and `protein_matched.py`.

The final outputs are under
`artifacts/paper_evidence/review_round4/transductive_references/`:

| Item | Output |
|---|---|
| Table 2, perturbation columns; S16 | `knockdown/evaluation_matched/report.json`, `auroc.csv`, `effects.csv`, `fills.csv` and `targets.csv`; additional comparator rows in `artifacts/paper_evidence/review_round3/comparators/evaluation/known_zeros/` |
| S17 | `knockdown/masked_f1/masked_f1_by_screen.json`, `masked_f1_report.json` |
| Table 2, sex columns; S18, Safe Fusion rows | `sex_zeros/evaluation/{pancreas,colon}/{auroc,fill_rates}.csv` |
| Figure 2 | `protein_cs/figures/pdl1_range_validation.png` and `.pdf` |
| S19 | `protein_cs/evaluation_cd274/replicate_association.csv`, `protein_cs/evaluation_cd274_raw_counts/continuous_protein_association.csv`, `protein_cs/evaluation/continuous_protein_association.csv` |
| S20 | `protein_cs/evaluation_within_state/{association,pdl1_pooled_rankings,paired_differences,global_fill}.csv` |

Table 2's DCA, scImpute and EnImpute perturbation rows use
`scripts/standard_imputers/evaluate_r3_known_zeros.py` from section 19. Sex-linked
comparator fits require the matched deployment fits of section 23; that section
finishes Table 2 and S18. The gene-panel availability rule excludes missing genes,
not missing donors from the bootstrap universe.

## 23. Downstream analyses and sex-linked comparators

Prerequisites: sections 19–22 and the deployment, thinning, disease and annotation
references in sections 8–14. First produce the standard-method references:

```bash
bash scripts/slurm_r3_downstream.sh submit
```

After completion, submit each following chain separately and wait for it:

```bash
bash scripts/slurm_r3_downstream_thinning.sh submit
bash scripts/slurm_r3_downstream_null.sh submit
sbatch --wait -p cs --mem=64G --time=03:00:00 scripts/slurm_r3_downstream_null.sh correlation recorded
sbatch --wait scripts/slurm_r3_downstream_table1.sh
bash scripts/analyses/transductive_downstream/reproduce_all.sh
```

`build_manifest.py` enumerates the masked, deployment, thinning and permuted
inputs. `pipeline.py` fits the teachers and values, `selector.py` produces
conditional and detection-weighted fills, and the original biological evaluators
compute the endpoints. `summarize.py`, `disease.py`, `disease_endpoints.py`,
`correlation.py` and `report.py` assemble the outputs.

Files below are relative to
`artifacts/paper_evidence/review_round4/transductive_downstream/`:

| Item | Output |
|---|---|
| Table 3 | `summary/thinning_reference.csv`, `summary/thinning_unit_values.csv` |
| S21 | `summary/endpoint_changes.csv`, `summary/s17_absolute.csv`, `summary/decomposition_counts.csv`, masked evaluation files |
| S22 | `summary/endpoint_changes.csv`, deployment evaluation files |
| S23 | `summary/disease_all.csv`, `disease/<tissue>_<fold>/deployment/disease_effects/<tissue>/{observed_summary,permutation_null}.csv` |
| S24 | `disease/<tissue>_<fold>/deployment/annotation/<tissue>/overall.csv` |
| S25 | `correlation/null/counts/{summary,versus_reference}.csv` |

Table 3 measures reduced error against independent unthinned counts. S22 measures
change from the recorded matrix. These are different endpoints. Disease tables
retain fold-specific gene panels; a fold without an estimable contrast stays NA.

After the downstream fits complete, reproduce the sex-linked comparators:

```bash
bash scripts/analyses/sex_zero_comparators/reproduce.sh
```

This submits fits, verification, evaluation and report in dependency order.
`fit.sh` runs the comparison methods; `evaluate.py` reuses the sex-zero kernel.
By default results go to
`artifacts/paper_evidence/review_round4/sex_zero_comparators/reproduction_run/`.
Table 2 and S18 read `evaluation/{pancreas,colon}/auroc.csv`; associated fill
rates read `fill_rates.csv`. The manuscript has unavailable EnImpute entries
where its saved fits did not cover every required fold. A fresh longer fit may
complete those cells; it does not change which values are printed in the manuscript.

## 24. Lupus regulatory-T-cell example

Prerequisites: the environments of section 1. Fetch and prepare the Perez and
Hao inputs before the fold pipeline. Each command below waits for completion.

```bash
sbatch --wait scripts/slurm_sle_fetch.sh
sbatch --wait scripts/slurm_sle_fetch_citeseq.sh
sbatch --wait scripts/slurm_sle_setup_env.sh
P=scripts/slurm_sle_pipeline.sh
sbatch --wait --array=0-2 --mem=96G --export=ALL,STAGE=prepare "$P"
sbatch --wait --array=0-8 --export=ALL,STAGE=build "$P"
sbatch --wait --array=0-47 --mem=48G --time=06:00:00 --export=ALL,STAGE=teachers "$P"
sbatch --wait --array=0-11 -p l40s_public --gres=gpu:l40s:1 --mem=48G --time=10:00:00 --export=ALL,STAGE=scvi_teacher "$P"
sbatch --wait --array=0-11 --export=ALL,STAGE=magic "$P"
sbatch --wait --array=0-11 -p l40s_public --gres=gpu:l40s:1 --mem=48G --export=ALL,STAGE=scvi "$P"
sbatch --wait --array=0-11 --mem=48G --time=03:00:00 --export=ALL,STAGE=stack "$P"
sbatch --wait --array=0-11 -p cs --mem=128G --time=10:00:00 --export=ALL,STAGE=selector "$P"
sbatch --wait --array=0-11 --export=ALL,STAGE=fills "$P"
sbatch --wait --array=0-6 --mem=128G --time=08:00:00 --export=ALL,STAGE=evaluate "$P"
bash scripts/analyses/sle_transductive/reproduce.sh
```

`sle_prepare.py`, `sle_prepare_citeseq.py` and `sle_build_fold.py` prepare the
sets and gene panels using `configs/sle_treg_genes.json`. The transductive
launcher reuses matching MAGIC/scVI fits, refits the other teachers, and runs
`sle_evaluate_zeros.py`, `sle_evaluate_focus_fills.py`, `sle_evaluate_clustering.py`,
`sle_evaluate_deploy.py`, `sle_evaluate_protein.py` and `sle_evaluate_masked.py`.

Table S29 reads the `conditional` and `conditional_detection` outputs under
`artifacts/paper_evidence/review_round4/sle_transductive/` and the corresponding
inductive files under `artifacts/paper_evidence/sle_treg_case/`:
`results/masked/paired_bootstrap.csv`, `results/q1_zeros_sex/per_gene.csv`,
`results/q3_clustering/treg_calls_summary.csv`, `results/q3_clustering/clustering_mean_over_folds.csv`,
`results/focus_fills/focus_fill_counts.csv`, `results/deploy_main/q4_effects.csv`,
`results/deploy_main/q4_null_de.csv`
and `results/q7_protein/fcrl3_protein_agreement.csv`. `comparisons/` contains the
same endpoints side by side; `ap_bootstrap.py` supplies masked AP intervals.
Hao supplies FcRL3 protein, not TLR5 protein, and is a healthy-donor reference.

## 25. Dropout-probability training target

Prerequisites: sections 20–22. This section supplies the dropout-probability
column of Table S9. It preserves the source score formula, seeds, calibration
and endpoint bootstraps. The manifest builders encode the matching input and
teacher paths; no saved manifest or fit is shipped.

```bash
O=artifacts/paper_evidence/review_round4/dropout_posterior
C=scripts/analyses/dropout_posterior
mkdir -p "$O/logs"
python "$C/build_manifest.py"
python "$C/extend_manifest.py"
sbatch --wait -A torch_pr_634_general -p cs --array=0-13,15-53%8 "$C/run.sh"
for dataset in Pancreas Colon CRISPRa; do
  sbatch --wait "$C/evaluate.sh" masked "$dataset"
done
for dataset in 'Colon 050' 'Colon 025' 'Pancreas 050'; do
  sbatch --wait "$C/evaluate.sh" thinning "$dataset"
done
sbatch --wait "$C/evaluate.sh" known adamson_crispri
for tissue in Pancreas Colon; do
  sbatch --wait -c 2 --mem=16G "$C/evaluate.sh" sex "$tissue"
done
sbatch --wait -c 2 --mem=16G "$C/evaluate.sh" protein
```

Table S9 reads `evaluation/masked/{Pancreas,Colon,CRISPRa}/results.csv`,
`evaluation/thinning/{Colon_050,Colon_025,Pancreas_050}/results.csv`, `evaluation/known/adamson_crispri/results.csv`,
`evaluation/sex/{Pancreas,Colon}/results.csv` and `evaluation/protein/results.csv`
under `$O`. Select the transductive primary posterior and its paired change
from the matching Safe Fusion chain. The excluded inductive Norman thinning
chain cannot reproduce its saved calibration selection on the tested hardware;
Table S9 does not require that endpoint. Missing dispersion limits the optional
negative-binomial sensitivity, not the primary Poisson posterior.

## 26. Scaling study

Prerequisites: sections 19 and 24's Perez download. No SLE model fits are needed.
`slurm_scale.sh submit` records new job IDs for resource accounting and accepts
`AFTER` to enforce stage dependencies. Prepare first:

```bash
sbatch --wait --array=0 --time=04:00:00 --mem=64G --export=ALL,STAGE=prepare scripts/slurm_scale.sh
bash scripts/slurm_scale.sh submit mask
```

After masking completes:

```bash
AFTER=mask bash scripts/slurm_scale.sh submit teachers
AFTER=mask bash scripts/slurm_scale.sh submit scvi_teacher
AFTER=mask bash scripts/slurm_scale.sh submit magic
AFTER=mask bash scripts/slurm_scale.sh submit scvi
AFTER=mask bash scripts/slurm_scale.sh submit alra
AFTER=mask bash scripts/slurm_scale.sh submit saver
AFTER="teachers scvi_teacher" bash scripts/slurm_scale.sh submit stack
AFTER=stack bash scripts/slurm_scale.sh submit selector
AFTER=scvi bash scripts/slurm_scale.sh submit scvi_probability
AFTER="magic scvi" bash scripts/slurm_scale.sh submit transductive_teachers
AFTER=transductive_teachers bash scripts/slurm_scale.sh submit transductive_stack
AFTER=transductive_stack bash scripts/slurm_scale.sh submit transductive_selector
AFTER="teachers scvi_teacher" bash scripts/slurm_scale.sh submit without_magic_stack
AFTER=without_magic_stack bash scripts/slurm_scale.sh submit without_magic_selector
```

Wait for all chains, then run:

```bash
sbatch --wait --array=0-11 --mem=160G --time=24:00:00 --export=ALL,STAGE=stack_replicate scripts/slurm_scale.sh
sbatch --wait --array=0-1 --time=02:00:00 --mem=32G --export=ALL,STAGE=check scripts/slurm_scale.sh
sbatch --wait --array=0 --mem=96G --time=08:00:00 --export=ALL,STAGE=evaluate scripts/slurm_scale.sh
sbatch --wait --array=0 --mem=16G --time=01:00:00 --export=ALL,STAGE=report scripts/slurm_scale.sh
```

Table S28 reads `results/paired_differences.csv` and
`results/{resources,safe_fusion_totals,accuracy_summary}.csv` under
`artifacts/paper_evidence/review_round3/scale/`. `scale_prepare.py` constructs
the nested samples, `scale_selector.py` computes the bounded-memory selector,
`scale_evaluate.py` computes accuracy and `scale_report.py` queries Slurm costs.
Exact wall time and sampled memory require the accounting records of the run;
new runs measure their own cost. No Slurm records are distributed by this update.

## 27. All-cell SVD, weighted kNN and ALRA

Prerequisites: sections 19–21, including the saved masked evaluations and all-cell
teachers. Run from the release root. The setup command below creates local code
links, output directories, a seven-unit manifest and source checksums. It never
imports data or results from another checkout. Existing mismatched links or
manifests cause an error. Run setup before each new analysis, then leave its
source unchanged until its jobs finish.

```bash
python scripts/analyses/setup_revision.py transductive_comparators
bash scripts/analyses/transductive_comparators/submit.sh
bash scripts/analyses/transductive_comparators/wait.sh
```

The launcher submits three parity checks, seven ALRA fits, three evaluations and
the report with `afterok` dependencies. SVD and weighted kNN reuse the all-cell
teachers. Table 1 and S5 read
`artifacts/paper_evidence/review_round4/transductive_comparators/paired_differences.csv`
and `absolute.csv`; select `Safe Fusion (transductive)` as the reference and
`SVD (transductive)`, `Weighted kNN (transductive)` and `ALRA (transductive)` as
comparators. Other rows retain section 21's outputs. Figure 1's curves are built
in section 29 after these fits complete.

Several final analyses compare numerical output with literal manuscript cells.
Supply the read-only final manuscript directory for these checks:

```bash
export SAFE_FUSION_MANUSCRIPT_DIR=/path/to/read-only/bioinformatics
```

This directory must contain `6_results.tex` and `supplementary_results.tex`.
It is read directly and is not copied into the repository. Two historical audits
also require the original, pre-update inputs below, under an external directory
called `REVISION_REFERENCES`. These are not distributed with the source release:

```text
round2_extras/part_a/table2_original_snapshot.csv
current_tables/original_supplementary_results.tex
```

Set `REVISION_REFERENCES` to that directory before sections 29–30. The setup
option `--reference-root "$REVISION_REFERENCES"` links only these read-only audit
inputs. The current supplement cannot replace the earlier supplement, and the
continuous-AUROC Table 2 cannot replace its historical snapshot. Missing inputs
stop the corresponding checks; no numerical or printed parity is assumed.

## 28. Label-aware baselines and sex-label sensitivity

Prerequisites: sections 21–23, complete sex-zero comparison fits including
EnImpute, and `SAFE_FUSION_MANUSCRIPT_DIR` from section 27. Section 23 writes to
`sex_zero_comparators/reproduction_run/`; these adapters use that same path.
The EnImpute launcher requests `06:00:00`. Do not use incomplete donor folds.

```bash
python scripts/analyses/setup_revision.py label_baselines
bash scripts/analyses/label_baselines/reproduce.sh
while squeue -u "$USER" -h -o %j | grep -q '^lb-'; do sleep 600; done
```

The launcher replays S16 and S18, checks their receipts, replays selectors,
checks the weighted AP implementation, evaluates masked baselines, runs the
supplementary all-cell label rule, and evaluates screen and tissue zeros.
Only the combined numerical gate releases new analyses. Its final report is
submitted after these stages; inspect `complete_job_accounting.txt` and require
successful completion of the report as well.

Paths below are relative to
`artifacts/paper_evidence/review_round4/label_baselines/`:

- Table 2 label baselines: `screens/baseline_per_target.csv`,
  `sex/<tissue>/label_rankings_auroc.csv`, and
  `sex/<tissue>/transductive_expected_count_auroc.csv`. Section 29 supplies the
  final continuous-score estimates and intervals for every Table 2 row.
- S6 accompanying text: `masked/<dataset>/report.json` and
  `masked/supplementary/<dataset>/report.json` (`Pancreas`, `Colon`, `adamson_crispri`,
  `papalexi_eccite`, `norman_crispra`).
- S16 corrections: `screens/verification_transductive/auroc.csv`,
  `screens/verification_inductive_comparators/knockdown/auroc.csv`,
  `screens/s16_paper_cells.csv`, and `screens/verification_receipt.json`.
- S18 and the colon sex-sensitivity text: `sex/colon/agreed25_auroc.csv`,
  `sex/colon/metadata34_auroc.csv`, `sex/colon/metadata_sex_audit.csv`, and
  `sex/colon/excluded_donors.csv`.

The strict fitting-cell donor baseline remains undefined for unseen donors.
The separately reported all-cell rule uses the donor teachers' fitting
population. The scripts retain that distinction and disclose printed rounding
differences separately from numerical replay.

## 29. Continuous AUROCs, all-cell curves and additional replicates

Prerequisites: sections 21, 23, 27 and 28. Supply the historical Table 2 snapshot
specified in section 27 and the final manuscript for the final audit.

```bash
python scripts/analyses/setup_revision.py round2_extras --reference-root "$REVISION_REFERENCES"
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/analyses/round2_extras/part_c/setup.py
bash scripts/analyses/round2_extras/reproduce.sh
bash scripts/analyses/round2_extras/wait.sh
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/plot_overview_f1.py --curves artifacts/paper_evidence/review_round4/round2_extras/part_b/selector_f1_fillrate_allcell_1000_points.csv --output artifacts/paper_evidence/review_round4/transductive_main/figures/f1_fillrate_3panel_oup.png
```

Part A checks the original source cells before four endpoint jobs and their
summary. Part B replaces only the all-cell SVD, weighted kNN and ALRA curves
and verifies the Table 1 points. Part C evaluates seed 1729, fits the four
additional seeds, evaluates them, and summarizes complete comparisons. The
final audit requires all comparisons; an `afterany` dependency does not treat a
failed fit as a completed comparison.

Table 2 reads `round2_extras/part_a/table2_aurocs.csv`: columns `continuous`,
`continuous_lower`, and `continuous_upper`, including the label-aware rows.
Figure 1 reads `round2_extras/part_b/selector_f1_fillrate_allcell_1000_points.csv`
and uses a 12 by 2.85 inch canvas. Table 1's highest-F1 count reads
`part_b/highest_f1.csv`. S4's additional block reads
`part_c/across_seeds.csv` and `part_c/paired_per_seed.csv`. All these paths are
under `artifacts/paper_evidence/review_round4/`.

## 30. Current value and fill-rule tables

Prerequisites: sections 9–11 and 19–23, including masked, recorded and thinning
teachers and selectors, and section 20's detection-weighted value reference.
Supply the earlier supplement snapshot from section 27. The old-input jobs are
parity prerequisites; the final tables use the current transductive outputs.
The sequential `--wait` commands below enforce the full dependency order.

```bash
python scripts/analyses/setup_revision.py current_tables --reference-root "$REVISION_REFERENCES"
C=scripts/analyses/current_tables
O=artifacts/paper_evidence/review_round4/current_tables
for part in masked recorded thinning; do
  sbatch --wait -A torch_pr_634_general -p cs -o "$O/logs/original-$part-%j.log" "$C/run.sh" original "$part"
done
sbatch --wait -A torch_pr_634_general -p cs -o "$O/logs/old-detection-%j.log" "$C/run.sh" extra old_detection
sbatch --wait -A torch_pr_634_general -p cs --array=0-3,5-8%5 -o "$O/logs/old-rule-%A_%a.log" "$C/old_rule.sh"
sbatch --wait -A torch_pr_634_general -p cs --array=4,9 -o "$O/logs/old-rule-norman-%A_%a.log" "$C/old_rule_norman.sh"
sbatch --wait -A torch_pr_634_general -p cs -o "$O/logs/old-print-%j.log" "$C/run.sh" original_print
sbatch --wait -A torch_pr_634_general -p cs --array=0-16%6 -o "$O/logs/linear-%A_%a.log" "$C/run.sh" linear
sbatch --wait -A torch_pr_634_general -p cs --array=0-16%5 -o "$O/logs/rule-%A_%a.log" "$C/run.sh" extra rule
sbatch --wait -A torch_pr_634_general -p cs -o "$O/logs/masked-%j.log" "$C/run.sh" masked
sbatch --wait -A torch_pr_634_general -p cs -o "$O/logs/recorded-%j.log" "$C/run.sh" extra recorded
sbatch --wait -A torch_pr_634_general -p cs -o "$O/logs/thinning-%j.log" "$C/run.sh" extra thinning
sbatch --wait -A torch_pr_634_general -p cs -o "$O/logs/report-%j.log" "$C/run.sh" report
```

The report pools rule counts and writes `current/rule/summary.csv`, comparison
LaTeX tables under `tables/`, and audit receipts. Paths below are relative to
`artifacts/paper_evidence/review_round4/current_tables/`:

| Item | Current output read |
|---|---|
| S12 | `current/masked/error_removed.csv`, `current/masked/log_error.csv` |
| S13 | `current/masked/error_removed.csv`, `current/masked/log_error_strata.csv` |
| S14 | `current/recorded/recorded_zero_fills.csv` |
| S15 | `current/thinning/bias.csv` |
| S26 | `current/rule/summary.csv`, `current/rule/{masked,recorded}/<unit>/calibration_report.json` |

The earlier autoencoder comparisons remain the inductive results of section 9;
the unchanged autoencoder runners do not accept the transductive teacher
contracts. Dixit is omitted from S14 because its complete transductive input
chain is unavailable. Neither case is replaced by a new fit or invented value.

## 31. Calibration, additional Table 3 methods and 5,000 genes

Prerequisites: section 23 for calibration and Table 3, and section 26 for the
scale experiment's input cohort and environments.

```bash
python scripts/analyses/setup_revision.py reviewer_extras
C=scripts/analyses/reviewer_extras
O=artifacts/paper_evidence/review_round4/reviewer_extras
sbatch --wait -A torch_pr_634_general -p cs -J extras-A -o "$O/logs/A-%j.log" "$C/job.sh" .venv/bin/python -u "$C/A_calibration/analyze.py"
bash "$C/B_thinning/submit.sh"
while squeue -u "$USER" -h -o %j | grep -q '^extras-B-'; do sleep 600; done
bash "$C/C_scale/code/submit.sh"
while squeue -u "$USER" -h -o %j | grep -q '^extras-C-'; do sleep 600; done
sbatch --wait -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 -o "$O/logs/final-%j.log" "$C/job.sh" bash "$C/finalize.sh"
```

The Table 3 chain first reproduces the existing bootstrap, then fits ALRA and
the single-method selectors on eight units, then summarizes. The scale chain
prepares 200,000 cells with 5,000 genes, masks, fits teachers and comparators,
fits fused values and selectors, and evaluates. The final report checks
completion. A missing method remains unavailable; inspect job states before
using the report.

Table 3 reads `B_thinning/all_rows.csv` and
`B_thinning/table3_new_rows_x100.csv`; the latter supplies ALRA and selector
columns on the manuscript's percentage scale. Section S8 calibration reads
`A_calibration/calibration.csv`. Its 5,000-gene text reads
`C_scale/results/method_summary.csv`, `C_scale/results/paired_differences.csv`
and the resource summaries produced by `C_scale/code/report.py`. All paths are
relative to `$O`. The default 2,000-gene S28 table still reads section 26.

## 32. Fill-fraction grid: Supplementary Figure S1

Prerequisites: sections 23 and 31, including all Table 3 methods and saved
1%/10% fills. Set `SAFE_FUSION_MANUSCRIPT_DIR` as in section 27.

```bash
python scripts/analyses/setup_revision.py fill_grid
bash scripts/analyses/fill_grid/submit.sh
while squeue -u "$USER" -h -o %j | grep -q '^fg-'; do sleep 600; done
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/analyses/fill_grid/verify.py
```

The chain runs eight unit gates, the numerical and printed Table 3 gate, eight
grid jobs, and the summary with `afterok` dependencies. It reuses saved scores
and values without refitting. The output root is
`artifacts/paper_evidence/review_round4/fill_grid/`. The figure reads
`fill_grid_long.csv`; `run.py summary` creates `fill_grid.pdf` and its PNG
preview `fill_grid.png`. Existing output destinations are exclusive.

## 33. Training-size sensitivity at 2,000 and 5,000 genes

Prerequisites: sections 26 and 31, including the completed default
200,000-cell transductive fits at both gene counts.

```bash
python scripts/analyses/setup_revision.py scale_caps
bash scripts/analyses/scale_caps/submit.sh
while squeue -u "$USER" -h -o %j | grep -q '^caps-'; do sleep 600; done
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/analyses/scale_caps/report.py
```

Both default value/selector replays must pass exact array checks before either
larger-cap chain starts. Only the two training caps change, from 600,000 to
6,000,000 value entries and from 2,000,000 to 20,000,000 selector candidates.
Section S8 reads `scale_caps/genes_<2000|5000>/evaluation/results/paired_differences.csv`
and `method_summary.csv`, relative to `artifacts/paper_evidence/review_round4/`.
Resource measurements are `scale_caps/resources.csv` and per-step
`*.resources.json`; a new run measures its own costs.

## 34. Additional comparison methods at scale

Prerequisites: sections 17, 19 and 26, including all four saved S28 evaluations,
R environments, the frozen scGPT checkpoint, and the final supplement for its
printed S28 gate. Run the monitor inside a Slurm allocation with enough time;
it uses `SLURM_JOB_END_TIME` and stops two hours before that allocation ends.

```bash
python scripts/analyses/setup_revision.py scale_comparators
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/analyses/scale_comparators/check_table.py
for n in 25000 50000 100000 200000; do
  .venv/bin/python scripts/analyses/scale_comparators/submit.py reproduce "$n"
done
bash scripts/analyses/scale_comparators/monitor.sh
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/analyses/scale_comparators/report.py
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 .venv/bin/python scripts/analyses/scale_comparators/check_results.py
```

The monitor waits for all four numerical reproductions before submitting DCA,
scGPT and the selectors. R methods first run at 25,000 cells; larger sizes are
eligible only when that run completes within 12 hours. Evaluations follow each
size's applicable methods. A monitor timeout is not completion: resume using
the existing ledger and require `DONE` and successful result checks.
Section S8 reads `scale_comparators/paired_differences.csv`,
`scale_comparators/method_summary.csv`, `scale_comparators/resources.csv` and
`scale_comparators/r_gates.json`, under `artifacts/paper_evidence/review_round4/`.
Table S28 itself retains the original section 26 rows.

## 35. Donor labels in the lupus example

Prerequisites: section 24, including both unlabeled transductive values,
inductive references, the CITE-seq inputs and the saved AP intervals.

```bash
python scripts/analyses/setup_revision.py sle_donor_labels
bash scripts/analyses/sle_donor_labels/reproduce.sh
bash scripts/analyses/sle_donor_labels/monitor.sh
```

The launcher builds input links, replays both unlabeled values, validates the
saved endpoints, fits twelve donor-label scVI teachers on L40S GPUs, then runs
corresponding CPU chains, both sets of evaluations, AP bootstrap, audits and
the final report. All dependencies preserve that order.

Section S9 and S29 read
`artifacts/paper_evidence/review_round4/sle_donor_labels/comparisons/`, including
`masked_f1_intervals.csv`, `q1_zeros_sex/per_gene.csv`,
`q3_clustering/treg_calls_summary.csv`,
`q3_clustering/clustering_mean_over_folds.csv`,
`focus_fills/focus_fill_counts.csv`, `deploy_main/q4_effects.csv`,
`deploy_main/q4_null_de.csv`, `deploy_main/q5_modules.csv`, and `q7_protein/fcrl3_protein_agreement.csv`.
The report also writes `masked_ap_intervals.csv`. The underlying donor-label
endpoints are retained under `conditional/results/` and
`conditional_detection/results/`; the inductive and unlabeled columns retain
section 24's references.

## Verification

Run this from the repository root on a compute node. It checks every Python file,
every shell launcher and each literal script path in this guide, including the
C++ AP helper and temporary-directory setup check. Git-listed files exclude
environments and generated outputs. The AP launcher compiles its shared library
and runs its numerical selfcheck before fitting.

```bash
srun -A torch_pr_634_general -p cs -c 2 --mem=8G -t 00:30:00 python - <<'PYCODE'
from pathlib import Path
import re, subprocess, sys
root = Path.cwd()
files = [root / p for p in subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], text=True).splitlines()]
python_files = sorted(p for p in files if p.suffix == ".py")
subprocess.run([sys.executable, "-m", "py_compile", *map(str, python_files)], check=True)
for path in files:
    if path.suffix in {".sh", ".s", ".sbatch"}:
        subprocess.run(["bash", "-n", str(path)], check=True)
subprocess.run(["g++", "-fopenmp", "-fsyntax-only", "scripts/analyses/label_baselines/masked/weighted_ap.cpp"], check=True)
subprocess.run([sys.executable, "scripts/analyses/test_setup_revision.py"], check=True)
paths = set(re.findall(r"(?:scripts|src|fusion)/[\w/.-]+\.(?:py|sh|s|R)", (root / "docs/REPRODUCTION.md").read_text()))
assert all((root / path).is_file() for path in paths)
print(f"Compiled {len(python_files)} Python files; shell syntax and {len(paths)} script paths passed.")
PYCODE
```
