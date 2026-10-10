#!/usr/bin/env bash
#SBATCH --job-name=sf-r3-setup
#SBATCH --account=torch_pr_634_general
#SBATCH --partition=cs
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-r3-setup-%x-%j.out
#SBATCH --error=logs/slurm-r3-setup-%x-%j.err

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
CONDA="${CONDA:-/scratch/yz5944/miniconda3/bin/conda}"
export CONDA_PKGS_DIRS="${TMPDIR:-/tmp}/sf-r3-conda-${SLURM_JOB_ID:-local}"
SOURCES=external_data/baselines/r3_comparators
LOCKS=scripts/standard_imputers/locks
mkdir -p "${SOURCES}" "${LOCKS}"

case "${METHOD:?METHOD must be dca, scimpute, screcover or enimpute}" in
  dca)
    [[ -x .conda-dca/bin/python ]] || "${CONDA}" create -y -p .conda-dca -c conda-forge python=3.8.20 pip
    .conda-dca/bin/pip install --no-cache-dir \
      tensorflow==2.4.4 keras==2.4.3 dca==0.3.4 numpy==1.19.5 h5py==2.10.0 \
      scanpy==1.7.2 anndata==0.7.6 pandas==1.2.5 scipy==1.6.3 scikit-learn==0.24.2 \
      numba==0.53.1 llvmlite==0.36.0 umap-learn==0.5.1 matplotlib==3.4.3 kopt==0.1.0 pyyaml==5.4.1
    .conda-dca/bin/python -c "import dca.api, tensorflow, keras, scanpy; print('dca ok', tensorflow.__version__, keras.__version__, scanpy.__version__)"
    .conda-dca/bin/pip freeze > "${LOCKS}/dca.txt"
    ;;
  scimpute)
    [[ -x .conda-scimpute/bin/R ]] || "${CONDA}" create -y -p .conda-scimpute -c conda-forge \
      r-base=4.3.3 r-penalized r-kernlab r-rsvd r-doparallel r-foreach r-matrix r-jsonlite
    curl -L --fail --max-time 180 \
      https://codeload.github.com/Vivianstats/scImpute/tar.gz/78556ee2dc0a9c830223b9e3e9f99387cd3e09a4 \
      -o "${SOURCES}/scImpute-78556ee.tar.gz"
    tar -xzf "${SOURCES}/scImpute-78556ee.tar.gz" -C "${SOURCES}"
    .conda-scimpute/bin/R CMD INSTALL --no-docs "${SOURCES}/scImpute-78556ee2dc0a9c830223b9e3e9f99387cd3e09a4"
    .conda-scimpute/bin/Rscript -e 'library(scImpute); cat("scImpute", as.character(packageVersion("scImpute")), "\n")'
    "${CONDA}" list -p .conda-scimpute > "${LOCKS}/scimpute.txt"
    ;;
  screcover)
    [[ -x .conda-screcover/bin/R ]] || "${CONDA}" create -y -p .conda-screcover -c conda-forge -c bioconda \
      bioconductor-screcover=1.26.0 r-jsonlite r-matrix
    .conda-screcover/bin/Rscript -e 'library(scRecover); cat("scRecover", as.character(packageVersion("scRecover")), "\n")'
    "${CONDA}" list -p .conda-screcover > "${LOCKS}/screcover.txt"
    ;;
  enimpute)
    bash scripts/standard_imputers/setup_enimpute.sh
    ;;
  *)
    echo "unknown METHOD ${METHOD}" >&2
    exit 2
    ;;
esac
