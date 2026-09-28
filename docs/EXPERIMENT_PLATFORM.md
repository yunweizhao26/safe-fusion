# Safe Fusion experimental platform

## Execution model

The Snakemake workflow runs:

`fetch -> validate -> preprocess -> split -> corrupt -> run methods -> standardize -> evaluate -> aggregate`

Splits use whole biological units. Corruption occurs before method fitting.
Method artifacts receive corrupted counts. Unmasked counts and hidden values
remain evaluation inputs. Run identifiers combine the Git revision and resolved
configuration hash.

## Method contract

Each standardized method directory contains:

- `mean.npy`: `float32` predictions in cells-by-genes order.
- `metadata.json`: method, parameters, seed, input hashes, and training split.
- Optional `variance.npy`, `fill_score.npy`, or `embedding.npy`.

Validation rejects shape or order mismatches, nonfinite values, negative
count-scale means, missing provenance, and contracts that declare held-out
data use during fitting. Each workflow rule records commands, hashes, seeds,
runtime, and resource use in `provenance/`.

## Environment

Use Python 3.11 or 3.12 from the repository root:

```bash
uv sync --python 3.11 --extra workflow --extra paper
source .venv/bin/activate
```

Run the synthetic workflow check:

```bash
snakemake --snakefile workflow/Snakefile --profile profiles/default
```

Run a paper configuration locally:

```bash
snakemake --snakefile workflow/Snakefile --profile profiles/default \
  --configfile configs/colon_pilot.yaml --cores 4
```

For Slurm, submit from the repository root:

```bash
snakemake --snakefile workflow/Snakefile --profile profiles/slurm
```

Supply account and partition values through the scheduler configuration.
For the launchers in `scripts/`, set `SBATCH_ACCOUNT` and, if needed,
`SBATCH_PARTITION` in the submitting shell (REPRODUCTION.md, Conventions).

## Scope

Synthetic configuration checks artifact and evaluation contracts. It does not
reproduce paper results. Paper runs require prepared public data, external
baseline environments, and configuration files listed in
[DATA.md](DATA.md) and [REPRODUCTION.md](REPRODUCTION.md).

The manuscript uses the workflow only to prepare the colon and pancreas
splits and masks. The method itself uses five teachers, the boosted fused
value and the MLP selector with `apply_topk` selection (REPRODUCTION.md,
section 3.2). The other selector architectures and budget modes of
`scripts/calibrated_selective_fill.py` are not used in the manuscript.
