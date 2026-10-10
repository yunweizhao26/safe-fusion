#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import io, sparse

def main() -> None:
    work, ncores, seed, ae_type = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    import dca
    import keras
    import scanpy
    import tensorflow
    from dca.api import dca as run_dca

    counts = sparse.csr_matrix(io.mmread(work / "counts.mtx").T, dtype=np.float32)
    n_cells, n_genes = counts.shape
    fitted = np.asarray(counts.sum(axis=0)).ravel() > 0
    adata = ad.AnnData(X=counts[:, fitted].toarray())
    started = time.perf_counter()
    run_dca(adata, ae_type=ae_type, return_info=True, random_state=seed, threads=ncores)
    elapsed = time.perf_counter() - started

    mean = np.zeros((n_cells, n_genes))
    mean[:, fitted] = np.asarray(adata.X, dtype=np.float64)
    mean.astype(np.float64).tofile(work / "imputed.bin")
    zero_inflated = ae_type.startswith("zinb")
    if zero_inflated:
        pi = np.zeros((n_cells, n_genes))
        theta = np.ones((n_cells, n_genes))
        pi[:, fitted] = np.asarray(adata.obsm["X_dca_dropout"], dtype=np.float64)
        theta[:, fitted] = np.asarray(adata.obsm["X_dca_dispersion"], dtype=np.float64)
        zero_probability = np.power(theta / (theta + mean), theta)
        posterior = pi / (pi + (1.0 - pi) * zero_probability)
        posterior[(counts.toarray() > 0) | ~fitted[None, :]] = 0.0
        posterior.astype(np.float64).tofile(work / "dropout.bin")
    report = {
        "method": "DCA" if not zero_inflated else "DCA (ZINB)",
        "package_version": dca.__version__ if hasattr(dca, "__version__") else "0.3.4",
        "source": "PyPI dca 0.3.4",
        "python": sys.version.split()[0],
        "dependencies": {"tensorflow": tensorflow.__version__, "keras": keras.__version__,
                         "scanpy": scanpy.__version__, "anndata": ad.__version__, "numpy": np.__version__},
        "settings": {
            "call": "dca.api.dca(adata, ae_type=ae_type, return_info=True, random_state=seed, threads=ncores)",
            "ae_type": ae_type,
            "defaults": {"mode": "denoise", "normalize_per_cell": True, "scale": True,
                         "log1p": True, "hidden_size": [64, 32, 64], "hidden_dropout": 0.0, "batchnorm": True,
                         "activation": "relu", "init": "glorot_uniform", "epochs": 300, "reduce_lr": 10,
                         "early_stop": 15, "batch_size": 32, "optimizer": "RMSprop"},
            "random_state": seed, "threads": ncores,
        },
        "genes_fitted": int(fitted.sum()),
        "genes_without_counts": int((~fitted).sum()),
        "dropout_probability": "posterior probability of the zero-inflation component given a zero count" if zero_inflated else None,
        "imputed_scale": "counts (mean of the fitted count distribution)",
        "elapsed_seconds": elapsed,
    }
    (work / "runner.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"DCA done {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
