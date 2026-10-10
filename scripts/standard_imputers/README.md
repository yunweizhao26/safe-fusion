# Standard imputers for Supplementary Table S1

`run_standard_imputers.py` runs SAUCIE, MAGIC, DeepImpute, scScope, scVI and
kNN smoothing on disease and tissue subsets of the Crohn's disease colon data
(CELLxGENE dataset `63ff2c52-cb63-44f0-bac3-d0b33373e312`). It keeps the imputer
functions and parameters of the original analysis:

- SAUCIE: 1,000 training steps with the package defaults.
- MAGIC: genes detected in fewer than five cells are removed, values are
  library-size normalized and square-root transformed, and MAGIC runs with
  t = 7 and 20 principal components. The output keeps only the modeled genes.
- DeepImpute: `MultiNet` defaults, `minVMR = 0.5`.
- scScope: 15 latent dimensions, masked loss, batch size 64, 500 epochs,
  T = 2, learning rate 1e-4, beta1 = 0.05.
- scVI: default model and training, with normalized expression as output.
- kNN smoothing: the mean of each cell's ten nearest cells (Euclidean
  distance, the cell included) on the recorded values.

Cells are restricted to `is_primary_data`, and every (disease, tissue) subset
is imputed separately from its recorded matrix `X`. The output of a subset is
written to `<output-root>/<method>/<dataset-id>/<disease>/<tissue>.npy`, and
existing outputs are kept. scScope runs for 500 epochs. This is the value in
the last version of the original script, which dates from the same time as
the scScope outputs of the original analysis. Earlier versions of the script
used 1,000 epochs.

## Environment

All six imputers run on CPU in one conda environment, `environment.yaml`. The
imputer packages have the versions of the original analysis, and the numerical
libraries are pinned to the build that produced the seeded outputs. The
launcher creates the environment at `.conda-standard-imputers` on first use.

| Imputer | Packages |
|---|---|
| SAUCIE | `vendor/SAUCIE`, TensorFlow 2.15 in `tf.compat.v1` graph mode, scikit-learn |
| MAGIC | `magic-impute` 3.0.0, scprep 1.2.3, graphtools 1.5.3 |
| DeepImpute | `vendor/deepimpute`, TensorFlow and Keras 2.15 |
| scScope | `vendor/scscope`, TensorFlow 2.15 in `tf.compat.v1` graph mode, phenograph 1.5.7 |
| scVI | scvi-tools 1.2.2.post2 |
| kNN smoothing | scikit-learn |

`vendor/` holds the copies of SAUCIE (KrishnaswamyLab/SAUCIE, MIT license),
DeepImpute 1.2 (lanagarmire/deepimpute, MIT license) and scScope 0.1.5
(AltschulerWu-Lab/scScope, Apache 2.0 license) used in the original analysis.
Their license files are included. These copies differ from the upstream code
by edits for TensorFlow 2 and by one seeding change:

- SAUCIE and scScope import `tensorflow.compat.v1` and disable TensorFlow 2
  behavior.
- SAUCIE layer names read `encoder_0` instead of `encoder0`, and the same
  change applies to the decoder layers.
- DeepImpute uses package-relative imports, enters `tf.init_scope()`, and
  falls back from the legacy Keras Adam optimizer.
- scScope `train` takes an optional `seed` and sets it as the seed of the
  TensorFlow graph that it builds. This change was added for the seeded runs.

The MAGIC code of the original analysis is identical to `magic-impute` 3.0.0.

## Reproducibility

All six imputers run on CPU: the launcher hides the GPUs
(`CUDA_VISIBLE_DEVICES=""`), and TensorFlow and PyTorch use deterministic
kernels. `--seed` (1729 in the launcher) seeds the Python, NumPy, TensorFlow
and PyTorch generators before each subset and starts a new TensorFlow graph,
so the output of a subset does not depend on the other subsets of the run.
The seed is also the graph seed of SAUCIE and scScope, the seed of
DeepImpute, the `random_state` of MAGIC and the scVI seed. kNN smoothing is
deterministic. Two runs of `slurm_standard_imputers.sh` in this environment
gave bit-identical outputs for all six imputers in the four subsets of
Table S1. The original analysis set no seed except the DeepImpute default of
1234.

## Commands

From the repository root:

```bash
jid=$(sbatch --parsable scripts/slurm_standard_imputers.sh)
sbatch --dependency=afterok:${jid} scripts/slurm_evaluate_fill_decisions.sh
```

The first launcher writes `artifacts/paper_evidence/standard_imputers/<method>/<dataset>/<disease>/<tissue>.npy`.
The second writes `artifacts/paper_evidence/fill_decisions/`, where
`table_s1.csv` holds Table S1.
