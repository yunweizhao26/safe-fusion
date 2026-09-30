#!/usr/bin/env bash

set -euo pipefail

SAFE_FUSION_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$SAFE_FUSION_ROOT"
CONDA="${CONDA:-/scratch/yz5944/miniconda3/bin/conda}"
ENV="${SAFE_FUSION_ROOT}/.conda-enimpute"
SRC="${SAFE_FUSION_ROOT}/external_data/baselines/r3_comparators"
export CONDA_PKGS_DIRS="${TMPDIR:-/tmp}/safefusion-enimpute-conda-$$"
export PYTHONNOUSERSITE=1
mkdir -p "$SRC"

if [ ! -x "$ENV/bin/R" ]; then
  "$CONDA" create -y -p "$ENV" -c conda-forge -c bioconda --override-channels \
    r-base=3.5.1 r-seurat=2.3.4=r351h29659fb_2 r-reticulate=1.15 r-rsvd=1.0.3 r-rspectra=0.16_0 \
    r-corpcor=1.6.9 r-penalized=0.9_51 r-kernlab=0.9_29 r-doparallel=1.0.15 r-foreach=1.5.0 \
    r-iterators=1.0.12 r-glmnet=2.0_18 r-lars=1.2 r-matrix=1.2_18 r-rcpp=1.0.4.6 r-ggplot2=3.3.0 \
    r-jsonlite=1.6.1 r-knitr=1.28 r-rmarkdown=2.1 python=3.8.15 pip sysroot_linux-64=2.17
fi

export PATH="$ENV/bin:$PATH"

"$ENV/bin/python" -m pip install --no-cache-dir \
  tensorflow==2.4.4 tensorflow-estimator==2.4.0 tensorboard==2.4.1 keras==2.4.3 \
  numpy==1.19.5 h5py==2.10.0 protobuf==3.20.3 six==1.15.0 \
  scipy==1.5.4 pandas==1.1.5 scikit-learn==0.24.2 anndata==0.7.5 scanpy==1.7.2 \
  numba==0.53.1 llvmlite==0.36.0 umap-learn==0.5.1 pynndescent==0.5.2 \
  matplotlib==3.3.4 seaborn==0.11.1 statsmodels==0.12.2 tables==3.6.1 \
  kopt==0.1.0 hyperopt==0.2.5 pyyaml==5.4.1 dca==0.3.4 \
  magic-impute==3.0.0 graphtools==1.5.3 scprep==1.2.3 tasklogger==1.2.0 pygsp==0.5.1

download() {
  [ -s "$SRC/$2" ] || curl -L --fail --max-time 300 "$1" -o "$SRC/$2"
}
download https://codeload.github.com/Zhangxf-ccnu/EnImpute/tar.gz/8e42911c0691a48ecf3985d9005fa0891c24330e EnImpute-8e42911c0691a48ecf3985d9005fa0891c24330e.tar.gz
download https://codeload.github.com/Vivianstats/scImpute/tar.gz/78556ee2dc0a9c830223b9e3e9f99387cd3e09a4 scImpute-78556ee2dc0a9c830223b9e3e9f99387cd3e09a4.tar.gz
download https://codeload.github.com/ChongC1990/scRMD/tar.gz/6a665334118f936a57eef9c83a3b31a26b4ffddd scRMD-6a665334118f936a57eef9c83a3b31a26b4ffddd.tar.gz
download https://cran.r-project.org/src/contrib/Archive/DrImpute/DrImpute_1.0.tar.gz DrImpute_1.0.tar.gz \
  || download https://cran.r-project.org/src/contrib/DrImpute_1.0.tar.gz DrImpute_1.0.tar.gz
download https://cran.r-project.org/src/contrib/Archive/SAVER/SAVER_1.1.2.tar.gz SAVER_1.1.2.tar.gz \
  || download https://cran.r-project.org/src/contrib/SAVER_1.1.2.tar.gz SAVER_1.1.2.tar.gz
download https://cran.r-project.org/src/contrib/Archive/Rmagic/Rmagic_2.0.3.tar.gz Rmagic_2.0.3.tar.gz

BUILD="${TMPDIR:-/tmp}/safefusion-enimpute-build-$$"
mkdir -p "$BUILD"
for name in EnImpute-8e42911c0691a48ecf3985d9005fa0891c24330e scImpute-78556ee2dc0a9c830223b9e3e9f99387cd3e09a4 scRMD-6a665334118f936a57eef9c83a3b31a26b4ffddd; do
  tar -xzf "$SRC/$name.tar.gz" -C "$BUILD"
done

installed() {
  "$ENV/bin/Rscript" -e "quit(status = as.integer(!(requireNamespace('$1', quietly = TRUE) && as.character(packageVersion('$1')) == '$2')))"
}
r_install() {
  installed "$1" "$2" || "$ENV/bin/R" CMD INSTALL --no-docs "$3"
}
r_install DrImpute 1.0 "$SRC/DrImpute_1.0.tar.gz"
r_install SAVER 1.1.2 "$SRC/SAVER_1.1.2.tar.gz"
r_install Rmagic 2.0.3 "$SRC/Rmagic_2.0.3.tar.gz"
r_install scImpute 0.0.9 "$BUILD/scImpute-78556ee2dc0a9c830223b9e3e9f99387cd3e09a4"
r_install scRMD 0.99.0 "$BUILD/scRMD-6a665334118f936a57eef9c83a3b31a26b4ffddd"
r_install EnImpute 1.1 "$BUILD/EnImpute-8e42911c0691a48ecf3985d9005fa0891c24330e/pkg"
rm -rf "$BUILD" "$CONDA_PKGS_DIRS"

export RETICULATE_PYTHON="$ENV/bin/python"
"$ENV/bin/Rscript" -e "suppressMessages(library(EnImpute)); for (p in c('EnImpute', 'Seurat', 'SAVER', 'scImpute', 'scRMD', 'DrImpute', 'Rmagic', 'rsvd', 'reticulate')) cat(p, as.character(packageVersion(p)), '\n'); cat('pymagic', reticulate::import('magic')[['__version__']], '\n')"
"$ENV/bin/dca" --help > /dev/null
echo ENIMPUTE_ENV_OK
