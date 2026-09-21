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

Supply account and partition values through scheduler configuration or
`sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION` when using a launcher.

## Scope

Synthetic configuration checks artifact and evaluation contracts. It does not
reproduce paper results. Paper runs require prepared public data, external
baseline environments, and configuration files listed in
[DATA.md](DATA.md) and [REPRODUCTION.md](REPRODUCTION.md).

The paper method uses MLP `apply_topk` selection. Logistic selection, dense
fusion, alternative selector architectures, and different budget modes are
ablations or sensitivity analyses.
