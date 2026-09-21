# Safe Fusion evidence reproduction

This file records commands used to produce the results in
[`PAPER_EVIDENCE.md`](PAPER_EVIDENCE.md). Run all commands from the repository
root. Create and activate the repository environment first:

```bash
uv sync --python 3.11 --extra workflow --extra paper
source .venv/bin/activate
```

Python 3.12 is also supported. Commands below use the activated environment.
Baselines with separate environments identify their requirements.

Paper commands fix split, bootstrap, and corruption seeds at 1729. Preserve
the environment lock and run provenance because numerical results can vary
across hardware and package builds. The primary paper artifacts
(`artifacts/paper_evidence/*_mask_primary.json`) are produced by
`scripts/locked_primary_analysis.py` from the run contracts and are unchanged
by any of the steps below.

## 1. Primary locked runs (colon, pancreatic islets, PBMC)

```bash
snakemake --snakefile workflow/Snakefile --profile profiles/default \
  --configfile configs/colon_pilot.yaml --cores 4
```

Repeat with `configs/pancreas_pilot.yaml` and `configs/pbmc_pilot.yaml`.
These DAGs fit every enabled method (including the dense Safe Fusion
implementation) and write contracts under
`artifacts/{colon,pancreas,pbmc}_runs/<run-id>/methods/`, then run
core/downstream/perturbation evaluation and the aggregate gate. The run ID
combines the git commit and the resolved configuration hash.

For the dense-fusion reconstruction comparison, replace `REPLACE_WITH_RUN_ID`
with the run directory printed by the workflow:

```bash
OMP_NUM_THREADS=2 python \
  scripts/locked_primary_analysis.py \
  --dataset colon_epithelial --corruption mask_010 \
  --coordinates artifacts/colon_runs/REPLACE_WITH_RUN_ID/data/colon_epithelial/coordinates/mask_010.parquet \
  --contracts-root artifacts/colon_runs/REPLACE_WITH_RUN_ID/methods/standardized/colon_epithelial/mask_010 \
  --baselines raw gene_mean gene_median random_fill svd_impute graph_smooth \
  --output artifacts/paper_evidence/colon_mask_primary.json
```

(Same for pancreas/pbmc; PBMC is descriptive because it has one test donor.)

## 2. Colon donor-level biology (uses existing contracts, no refit)

Generate MLP contracts with section 6 before running this evaluation. Dense
Safe Fusion remains a reconstruction baseline.

```bash
COLON=artifacts/colon_runs/REPLACE_WITH_RUN_ID
M=$COLON/methods
OMP_NUM_THREADS=2 python \
  scripts/evaluate_colon_donor_biology.py \
  --truth external_data/prepared/colon_epithelial.h5ad \
  --corrupted $COLON/data/colon_epithelial/corrupted/mask_010.h5ad \
  --coordinates $COLON/data/colon_epithelial/coordinates/mask_010.parquet \
  --splits $COLON/data/colon_epithelial/splits.parquet \
  --method graph_smooth=$M/standardized/colon_epithelial/mask_010/graph_smooth \
  --method safe_fusion=$M/standardized/colon_epithelial/mask_010/safe_fusion \
  --method safe_fusion_mlp_020=artifacts/paper_evidence/selector_mlp_range/colon/safe_fusion_calibrated_mlp_topk_0p02 \
  --method safe_fusion_mlp_050=artifacts/paper_evidence/selector_mlp_range/colon/safe_fusion_calibrated_mlp_topk_0p05 \
  --method safe_fusion_mlp_081=artifacts/paper_evidence/selector_mlp_range/colon/safe_fusion_calibrated_mlp_topk_0p081 \
  --method safe_fusion_mlp_100=artifacts/paper_evidence/selector_mlp_range/colon/safe_fusion_calibrated_mlp_topk_0p1 \
  --output-dir artifacts/paper_evidence/colon_donor_biology/results \
  --bootstrap 2000 --seed 1729
```

## 3. Pancreatic-islet donor cross-fit (dense + selective)

```bash
PAN=artifacts/pancreas_runs/REPLACE_WITH_RUN_ID
CORRUPTED=$PAN/data/pancreas_islets/corrupted/mask_010.h5ad
COORDS=$PAN/data/pancreas_islets/coordinates/mask_010.parquet
CF=artifacts/paper_evidence/pancreas_crossfit

python scripts/make_pancreas_crossfit_splits.py \
  --input external_data/prepared/pancreas_islets.h5ad \
  --output-dir $CF --folds 3 --seed 1729

for fold in 0 1 2; do
  python scripts/run_leakage_safe_method.py --method graph_smooth \
    --input $CORRUPTED --coordinates $COORDS \
    --splits $CF/fold_$fold/splits.parquet \
    --output $CF/fold_$fold/graph_smooth --seed 1729
  python scripts/run_leakage_safe_method.py --method safe_fusion \
    --input $CORRUPTED --coordinates $COORDS \
    --splits $CF/fold_$fold/splits.parquet \
    --output $CF/fold_$fold/safe_fusion_b32 --seed 1729 --epochs 12 --batch-size 32
done

for spec in "020 0.02" "050 0.05" "081 0.081" "100 0.10"; do
  set -- $spec
  for fold in 0 1 2; do
    python scripts/derive_sparse_safe_fusion.py \
      --input $CORRUPTED --splits $CF/fold_$fold/splits.parquet \
      --fusion-contract $CF/fold_$fold/safe_fusion_b32 \
      --teacher-contract $CF/fold_$fold/graph_smooth \
      --output $CF/fold_$fold/safe_fusion_selective_$1 \
      --confidence-quantile 0.90 --fill-budget $2 --fallback raw \
      --method-name safe_fusion_selective_$1
  done
done

OMP_NUM_THREADS=2 python \
  scripts/evaluate_pancreas_crossfit_biology.py \
  --truth external_data/prepared/pancreas_islets.h5ad \
  --corrupted $CORRUPTED --coordinates $COORDS \
  --crossfit-dir $CF --safe-fusion-subdir safe_fusion_b32 \
  --extra-method safe_fusion_selective_020=safe_fusion_selective_020 \
  --extra-method safe_fusion_selective_050=safe_fusion_selective_050 \
  --extra-method safe_fusion_selective_081=safe_fusion_selective_081 \
  --extra-method safe_fusion_selective_100=safe_fusion_selective_100 \
  --output-dir artifacts/paper_evidence/pancreas_crossfit_selective/results \
  --folds 3 --bootstrap 2000 --seed 1729
```

## 4. Norman CRISPRa perturbation benchmark

Requires the GEARS-processed Norman file with its raw UMI counts layer. Set
`NORMAN_INPUT` to that file; see [`DATA.md`](DATA.md) for the public source.

```bash
N=artifacts/paper_evidence/norman_crispra
: "${NORMAN_INPUT:?set NORMAN_INPUT to perturb_processed.h5ad}"

python scripts/prepare_norman_crispra.py \
  --input "$NORMAN_INPUT" \
  --output external_data/prepared/norman_crispra.h5ad \
  --seed 1729 --max-cells-per-condition 60 --max-control-cells 1000 \
  --min-cells-per-condition 60 --target-log2fc-min 0.5 --target-p-max 0.05

python scripts/setup_norman_crispra_experiment.py \
  --input external_data/prepared/norman_crispra.h5ad \
  --output-dir $N --test-fraction 0.30 --mask-fraction 0.10 --seed 1729

python scripts/run_leakage_safe_method.py --method graph_smooth \
  --input $N/corrupted.h5ad --coordinates $N/coordinates.parquet \
  --splits $N/splits.parquet --output $N/methods/graph_smooth --seed 1729

python scripts/run_leakage_safe_method.py --method safe_fusion \
  --input $N/corrupted.h5ad --coordinates $N/coordinates.parquet \
  --splits $N/splits.parquet --output $N/methods/safe_fusion \
  --seed 1729 --epochs 12 --batch-size 64

python scripts/derive_sparse_safe_fusion.py \
  --input $N/corrupted.h5ad --splits $N/splits.parquet \
  --fusion-contract $N/methods/safe_fusion \
  --teacher-contract $N/methods/graph_smooth \
  --output $N/methods/safe_fusion_selective_020 \
  --confidence-quantile 0.90 --fill-budget 0.02 --fallback raw \
  --method-name safe_fusion_selective_020

OMP_NUM_THREADS=2 python \
  scripts/evaluate_perturbation_preservation.py \
  --truth external_data/prepared/norman_crispra.h5ad \
  --corrupted $N/corrupted.h5ad --splits $N/splits.parquet \
  --method graph_smooth=$N/methods/graph_smooth \
  --method safe_fusion=$N/methods/safe_fusion \
  --method safe_fusion_selective_020=$N/methods/safe_fusion_selective_020 \
  --output-dir $N/results --bootstrap 2000 --seed 1729
```

## 5. Publication gate regeneration

The exact aggregate command for each pilot run is stored in
`artifacts/*_runs/<run-id>/provenance/aggregate.json`. Rerun it to regenerate
`tables/consolidated_metrics.parquet` and
`tables/publication_gate.json`:

```bash
cmd=$(python -c "import json; print(' '.join(json.load(open('artifacts/colon_runs/REPLACE_WITH_RUN_ID/provenance/aggregate.json'))['command']))")
eval "$cmd"
```

The gate selects the baseline per (dataset, corruption) on validation units
only, evaluates on test units only, requires ≥2 test units, and uses
cell-count-weighted pooled log1p MAE with a weighted paired donor bootstrap —
the same protocol as the primary endpoint.

## 6. MLP selective filling

The manuscript selector is an MLP fitted on known masked positives in the
development or validation split. `apply_topk` ranks unlabeled held-out scores
and fills the requested fraction. Held-out labels do not fit the MLP or choose
entries.

```bash
PY="python"
COL=artifacts/colon_runs/REPLACE_WITH_RUN_ID
PAN=artifacts/pancreas_runs/REPLACE_WITH_RUN_ID
CF=artifacts/paper_evidence/pancreas_crossfit

# Colon locked split (calibrate on validation, apply to test)
$PY scripts/calibrated_selective_fill.py \
  --corrupted $COL/data/colon_epithelial/corrupted/mask_010.h5ad \
  --truth $COL/data/colon_epithelial/preprocessed.h5ad \
  --coordinates $COL/data/colon_epithelial/coordinates/mask_010.parquet \
  --splits $COL/data/colon_epithelial/splits.parquet \
  --fusion-contract $COL/methods/standardized/colon_epithelial/mask_010/safe_fusion \
  --teacher-contract $COL/methods/standardized/colon_epithelial/mask_010/graph_smooth \
  --teacher-contract $COL/methods/standardized/colon_epithelial/mask_010/svd_impute \
  --teacher-contract $COL/methods/standardized/colon_epithelial/mask_010/gene_median \
  --output-dir artifacts/paper_evidence/selector_mlp_range/colon \
  --architecture mlp --budget-mode apply_topk \
  --budgets 0.02 0.05 0.081 0.10 --fit-split validation --fit-cells 3852 --seed 1729

# Pancreas cross-fit (calibrate on development per fold, apply to held-out donors)
for fold in 0 1 2; do
  $PY scripts/calibrated_selective_fill.py \
    --corrupted $PAN/data/pancreas_islets/corrupted/mask_010.h5ad \
    --truth $PAN/data/pancreas_islets/preprocessed.h5ad \
    --coordinates $PAN/data/pancreas_islets/coordinates/mask_010.parquet \
    --splits $CF/fold_$fold/splits.parquet \
    --fusion-contract $CF/fold_$fold/safe_fusion_b32 \
    --teacher-contract $CF/fold_$fold/graph_smooth \
    --output-dir $CF/fold_$fold/selector_mlp_range --budgets 0.02 0.05 0.081 \
    --architecture mlp --budget-mode apply_topk \
    --fit-split development --seed 1729
done

# CRISPRa (calibrate on development, apply to test)
$PY scripts/calibrated_selective_fill.py \
  --corrupted artifacts/paper_evidence/norman_crispra/corrupted.h5ad \
  --truth external_data/prepared/norman_crispra.h5ad \
  --coordinates artifacts/paper_evidence/norman_crispra/coordinates.parquet \
  --splits artifacts/paper_evidence/norman_crispra/splits.parquet \
  --fusion-contract artifacts/paper_evidence/norman_crispra/methods/safe_fusion \
  --teacher-contract artifacts/paper_evidence/norman_crispra/methods/graph_smooth \
  --output-dir artifacts/paper_evidence/selector_mlp_range/norman_crispra \
  --architecture mlp --budget-mode apply_topk \
  --budgets 0.02 0.05 --fit-split development --max-fit-rows 2000000 --seed 1729
```

Then rerun the corresponding evaluators with the MLP contracts as additional
methods. See sections 2 through 4.

## Matched-budget selector/value attribution

Colon template:

```bash
$PY scripts/selector_attribution.py \
  --corrupted $COL/data/colon_epithelial/corrupted/mask_010.h5ad \
  --truth $COL/data/colon_epithelial/preprocessed.h5ad \
  --coordinates $COL/data/colon_epithelial/coordinates/mask_010.parquet \
  --splits $COL/data/colon_epithelial/splits.parquet \
  --fusion-contract $COL/methods/standardized/colon_epithelial/mask_010/safe_fusion \
  --teacher-contract $COL/methods/standardized/colon_epithelial/mask_010/graph_smooth \
  --teacher-contract $COL/methods/standardized/colon_epithelial/mask_010/svd_impute \
  --teacher-contract $COL/methods/standardized/colon_epithelial/mask_010/gene_median \
  --output-dir artifacts/paper_evidence/selector_attribution/colon \
  --fit-split validation --unit-column donor --budgets 0.02 \
  --bootstrap 2000 --seed 1729
```

For pancreas, run the same command once per fold with the fold splits,
`safe_fusion_b32`, `graph_smooth`, `--fit-split development`, and
`--unit-column donor`; combine the three disjoint outputs with
`combine_selector_attribution.py`. For CRISPRa use the Norman paths above,
`--fit-split development`, and `--unit-column target`.

Architecture screening holds features and fill values fixed. Add:

```bash
--variants full \
--architectures logistic hist_gbdt extra_trees mlp \
--rank-ensemble --budgets 0.02 0.05 0.10 --max-fit-rows 400000
```

The retained CPU array is `scripts/slurm_selector_architectures.sh`. The
FT-Transformer sensitivity uses `scripts/slurm_selector_ft_transformer.sh`
with its baseline environment.

Run the manuscript selector as:

```bash
$PY scripts/calibrated_selective_fill.py ... \
  --architecture mlp --budget-mode apply_topk \
  --budgets 0.02 0.05 0.10
```

`apply_topk` does not use target labels, but the target score quantile sets the
cutoff; its metadata therefore records `test_values_used_for_thresholds: true`
and `test_labels_used_for_thresholds: false`. Logistic
`calibration_threshold` is the fixed-threshold ablation. Retained deployment
and biology arrays are
`scripts/slurm_selector_{deployment,biology,exact_budget,exact_budget_biology}.sh`.

## Post-facto markers and PBMC protein

```bash
$PY scripts/evaluate_postfacto_markers.py \
  --truth $COL/data/colon_epithelial/preprocessed.h5ad \
  --corrupted $COL/data/colon_epithelial/corrupted/mask_010.h5ad \
  --subset-splits $COL/data/colon_epithelial/splits.parquet \
  --method calibrated_0p02=artifacts/paper_evidence/colon_calibrated/safe_fusion_calibrated_0p02 \
  --method graph_smooth=$COL/methods/standardized/colon_epithelial/mask_010/graph_smooth \
  --method safe_fusion_dense=$COL/methods/standardized/colon_epithelial/mask_010/safe_fusion \
  --output-dir artifacts/paper_evidence/postfacto_markers/colon \
  --label-column cell_type --donor-column donor --bootstrap 2000 --seed 1729

$PY scripts/evaluate_postfacto_markers.py \
  --truth $PAN/data/pancreas_islets/preprocessed.h5ad \
  --corrupted $PAN/data/pancreas_islets/corrupted/mask_010.h5ad \
  --method calibrated_fixed_0p02=artifacts/paper_evidence/pancreas_crossfit_calibrated_fixed/results/oof_calibrated_fixed_0p02.npy \
  --method graph_smooth=artifacts/paper_evidence/pancreas_crossfit_calibrated_fixed/results/oof_graph_smooth.npy \
  --method safe_fusion_dense=artifacts/paper_evidence/pancreas_crossfit_calibrated_fixed/results/oof_safe_fusion.npy \
  --output-dir artifacts/paper_evidence/postfacto_markers/pancreas \
  --label-column cell_type --donor-column donor --stratify-column condition \
  --bootstrap 2000 --seed 1729
```

PBMC calibration and protein evaluation:

```bash
PB=artifacts/pbmc_runs/REPLACE_WITH_RUN_ID
PBM=$PB/methods/standardized/pbmc/mask_010

$PY scripts/calibrated_selective_fill.py \
  --corrupted $PB/data/pbmc/corrupted/mask_010.h5ad \
  --truth $PB/data/pbmc/preprocessed.h5ad \
  --coordinates $PB/data/pbmc/coordinates/mask_010.parquet \
  --splits $PB/data/pbmc/splits.parquet \
  --fusion-contract $PB/methods/rerun/pbmc/mask_010/safe_fusion \
  --teacher-contract $PBM/graph_smooth \
  --teacher-contract $PBM/svd_impute \
  --teacher-contract $PBM/gene_median \
  --output-dir artifacts/paper_evidence/pbmc_calibrated \
  --budgets 0.02 0.05 --fit-split validation --seed 1729

$PY scripts/evaluate_protein_biology.py \
  --truth $PB/data/pbmc/preprocessed.h5ad \
  --corrupted $PB/data/pbmc/corrupted/mask_010.h5ad \
  --coordinates $PB/data/pbmc/coordinates/mask_010.parquet \
  --splits $PB/data/pbmc/splits.parquet \
  --contracts-root $PBM \
  --methods corrupted_raw=$PBM/raw graph_smooth=$PBM/graph_smooth \
    dense_safe_fusion=$PB/methods/rerun/pbmc/mask_010/safe_fusion \
    calibrated_0p02=artifacts/paper_evidence/pbmc_calibrated/safe_fusion_calibrated_0p02 \
    calibrated_0p05=artifacts/paper_evidence/pbmc_calibrated/safe_fusion_calibrated_0p05 \
  --output artifacts/paper_evidence/pbmc_calibrated_protein.parquet \
  --summary artifacts/paper_evidence/pbmc_calibrated_protein.json --seed 1729
```

## External CRISPRi, KO, and ECCITE-seq screens

The source files are the harmonized raw-count H5ADs from scPerturb Zenodo
record 10044268. The retained source hashes are in each preparation report.

```bash
for spec in \
  "adamson_crispri AdamsonWeissman2016_GSM2406681_10X010.h5ad" \
  "dixit_ko DixitRegev2016.h5ad" \
  "papalexi_eccite PapalexiSatija2021_eccite_RNA.h5ad"; do
  set -- $spec
  DATASET=$1
  SOURCE=$2
  $PY scripts/prepare_external_perturbseq.py \
    --input external_data/scperturb/$SOURCE \
    --output external_data/prepared/$DATASET.h5ad --dataset $DATASET
  $PY scripts/setup_norman_crispra_experiment.py \
    --input external_data/prepared/$DATASET.h5ad \
    --output-dir artifacts/external_perturbseq/$DATASET \
    --split-column preassigned_split
  $PY scripts/run_leakage_safe_method.py --method graph_smooth \
    --input artifacts/external_perturbseq/$DATASET/corrupted.h5ad \
    --coordinates artifacts/external_perturbseq/$DATASET/coordinates.parquet \
    --splits artifacts/external_perturbseq/$DATASET/splits.parquet \
    --output artifacts/external_perturbseq/$DATASET/graph_smooth
  $PY scripts/run_leakage_safe_method.py --method safe_fusion \
    --input artifacts/external_perturbseq/$DATASET/corrupted.h5ad \
    --coordinates artifacts/external_perturbseq/$DATASET/coordinates.parquet \
    --splits artifacts/external_perturbseq/$DATASET/splits.parquet \
    --output artifacts/external_perturbseq/$DATASET/safe_fusion \
    --epochs 12 --batch-size 64
  $PY scripts/calibrated_selective_fill.py \
    --corrupted artifacts/external_perturbseq/$DATASET/corrupted.h5ad \
    --truth external_data/prepared/$DATASET.h5ad \
    --coordinates artifacts/external_perturbseq/$DATASET/coordinates.parquet \
    --splits artifacts/external_perturbseq/$DATASET/splits.parquet \
    --fusion-contract artifacts/external_perturbseq/$DATASET/safe_fusion \
    --teacher-contract artifacts/external_perturbseq/$DATASET/graph_smooth \
    --output-dir artifacts/external_perturbseq/$DATASET/calibrated \
    --budgets 0.02 --fit-split development
  $PY scripts/evaluate_perturbation_preservation.py \
    --truth external_data/prepared/$DATASET.h5ad \
    --corrupted artifacts/external_perturbseq/$DATASET/corrupted.h5ad \
    --splits artifacts/external_perturbseq/$DATASET/splits.parquet \
    --method graph_smooth=artifacts/external_perturbseq/$DATASET/graph_smooth \
    --method safe_fusion_dense=artifacts/external_perturbseq/$DATASET/safe_fusion \
    --method safe_fusion_calibrated_2pct=artifacts/external_perturbseq/$DATASET/calibrated/safe_fusion_calibrated_0p02 \
    --output-dir artifacts/external_perturbseq/$DATASET/evaluation
done
```

## Zebrafish trajectory

```bash
$PY scripts/prepare_zebrafish_trajectory.py \
  --input external_data/trajectory/zebrafish_embryogenesis_axial_mesoderm.h5ad \
  --output external_data/prepared/zebrafish_trajectory.h5ad
$PY scripts/setup_norman_crispra_experiment.py \
  --input external_data/prepared/zebrafish_trajectory.h5ad \
  --output-dir artifacts/external_trajectory/zebrafish \
  --split-column preassigned_split
# Run graph_smooth, safe_fusion, and calibrated_selective_fill.py as above,
# replacing the root with artifacts/external_trajectory/zebrafish.
$PY scripts/evaluate_trajectory_preservation.py \
  --truth external_data/prepared/zebrafish_trajectory.h5ad \
  --corrupted artifacts/external_trajectory/zebrafish/corrupted.h5ad \
  --splits artifacts/external_trajectory/zebrafish/splits.parquet \
  --method graph_smooth=artifacts/external_trajectory/zebrafish/graph_smooth \
  --method safe_fusion_dense=artifacts/external_trajectory/zebrafish/safe_fusion \
  --method safe_fusion_calibrated_2pct=artifacts/external_trajectory/zebrafish/calibrated/safe_fusion_calibrated_0p02 \
  --output-dir artifacts/external_trajectory/zebrafish/evaluation
```

## Complete five-task real-data downstream benchmark

The complete benchmark uses donor-held-out pancreas and colon data, held-out
zebrafish cells, and real
Norman CRISPRa, Adamson CRISPRi, Dixit knockout, and Papalexi ECCITE-seq
interventions. SERGIO is not used. The selector is evaluated at every integer
selected-zero fraction from 1 through 10%. Run all compute through Slurm:

```bash
prepare_job=$(sbatch --parsable scripts/slurm_prepare_complete_downstream_teachers.sh)
selector_job=$(sbatch --parsable --dependency=afterok:${prepare_job} scripts/slurm_complete_downstream_selectors.sh)
evaluate_job=$(sbatch --parsable --dependency=afterok:${selector_job} scripts/slurm_evaluate_complete_downstream.sh)
sbatch --dependency=afterok:${evaluate_job} scripts/slurm_finalize_complete_downstream.sh
```

The finalizer writes the combined tables and five-panel figure under
`artifacts/paper_evidence/downstream_complete/summary/` and fails unless all
five tasks, all ten selected fractions, all 24 pancreas donors, all 9 colon
donors, all 12 trajectory stages, and all 128 real perturbation regulators are
present and pass the leakage checks.

## Official scGCL adapter

The official repository is pinned at commit
`317015acdf06d2929c20a7d2858bac539b3d8ebd`.

The compatibility layer preserves upstream median-library normalization,
Scanpy HVG selection, the undirected unit-weight 15-NN graph, the AFGRL
encoder/teacher and neighbor implementation, and the original summed ZINB
objective. It changes only input/output plumbing and device-safe tensor
operations, including an equivalent neighbor-index implementation that avoids
upstream's CPU/GPU indexing mismatch and device-local detached mean/dispersion
clamps that preserve upstream's effective gradients; labels and masked truth
never enter fitting. Because upstream
exports an L2-normalized log-expression embedding, the count-scale contract
restores the corrupted input L2 magnitude, applies `expm1`, and preserves the
selected-gene library mass; both the official embedding and reconstructed
log1p matrix are retained for audit.

Install the scGCL baseline environment and set the input run directories in
`scripts/sbatch_scgcl_baselines.s`. Submit it from the repository root with
your cluster's account and partition. The
adapter writes atomic optimizer/model/RNG checkpoints every 25 epochs, and the
batch script resumes them after preemption. After each fit it automatically
runs `scripts/evaluate_scgcl_baseline.py` on the locked test mask and writes a
paired biological-unit bootstrap report. The pinned default learning rate
(`1e-3`) becomes non-finite at epoch 68 on pancreas under both the
CUDA-compatible path and upstream-native CPU execution; `1e-4` delays the same
failure to epoch 165. At `1e-5`, pancreas completes but colon becomes
non-finite after epoch 240. The retained common-setting 300-epoch result is
therefore run as an explicit post-default numerical-stability sensitivity at
`1e-6`, with every other pinned setting unchanged. Metadata and the evaluator
retain the default, effective learning rate, and complete failure note. The
launcher follows the upstream trainer's CPU execution. This sensitivity must
not be described as untouched standard usage.

## Environment notes

- The Norman source file is read with bounded h5py block reads; do not open it
  with AnnData backed fancy indexing (it materializes the full dense matrix).
- `artifacts/` is git-ignored; the scripts, configs, and docs are the
  reproducible surface. Raw data and the prepared h5ads are large and are not
  committed.
