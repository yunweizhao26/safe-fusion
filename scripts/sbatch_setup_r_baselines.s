#!/bin/bash
#SBATCH --job-name=sf_r_base
#SBATCH --output=logs/sf_r_base_%j.log
#SBATCH --error=logs/sf_r_base_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G

# Build a conda R environment with ALRA and SAVER for the official
# zero-preserving and uncertainty-aware baselines.

set -e
SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"

CONDA=$(command -v conda)
export CONDA_PKGS_DIRS="${TMPDIR:-/tmp}/safefusion-conda"
export R_LIBS="$PWD/.conda-r-baselines/lib/R/library"
mkdir -p external_data/baselines

if [ ! -x .conda-r-baselines/bin/R ]; then
  "$CONDA" create -y -p .conda-r-baselines -c conda-forge \
    r-base=4.3.3 r-irlba r-matrix r-rcpp r-jsonlite r-devtools r-remotes \
    r-glmnet r-foreach r-iterators r-doparallel r-rcpparmadillo r-rsvd
else
  "$CONDA" install -y -p .conda-r-baselines -c conda-forge \
    r-irlba r-matrix r-rcpp r-jsonlite r-devtools r-remotes \
    r-glmnet r-foreach r-iterators r-doparallel r-rcpparmadillo r-rsvd
fi

cd external_data/baselines
curl -L --fail --max-time 180 https://codeload.github.com/KlugerLab/ALRA/zip/refs/heads/master -o ALRA.zip
unzip -q -o ALRA.zip
curl -L --fail --max-time 180 https://codeload.github.com/mohuangx/SAVER/zip/refs/heads/master -o SAVER.zip
unzip -q -o SAVER.zip
cd "$SAFE_FUSION_ROOT"

.conda-r-baselines/bin/R CMD INSTALL --no-docs external_data/baselines/SAVER-master
.conda-r-baselines/bin/R CMD INSTALL --no-docs external_data/baselines/ALRA-master

echo R_BASELINES_OK
.conda-r-baselines/bin/Rscript -e "library(SAVER); cat('SAVER ok\n')"
.conda-r-baselines/bin/Rscript -e "library(ALRA); cat('ALRA ok\n')"
