#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrupted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--resolution", type=float, default=1.0)
    args = parser.parse_args()

    source = ad.read_h5ad(args.corrupted)
    counts = source.layers["corrupted_counts"]
    counts = counts.astype(np.float32) if sparse.issparse(counts) else sparse.csr_matrix(np.asarray(counts, dtype=np.float32))
    adata = ad.AnnData(X=counts)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=args.n_pcs, random_state=args.seed)
    sc.pp.neighbors(adata, n_neighbors=args.n_neighbors, n_pcs=args.n_pcs, random_state=args.seed)
    sc.tl.leiden(adata, resolution=args.resolution, random_state=args.seed, flavor="igraph", n_iterations=2, directed=False)
    sizes = adata.obs["leiden"].value_counts().sort_index()
    result = {
        "kcluster": int(len(sizes)),
        "cluster_sizes": {str(key): int(value) for key, value in sizes.items()},
        "input": str(args.corrupted),
        "rule": "number of Leiden clusters of the input (normalize_total 1e4, log1p, PCA, kNN graph, Leiden)",
        "settings": {
            "n_pcs": args.n_pcs, "n_neighbors": args.n_neighbors, "resolution": args.resolution,
            "leiden_flavor": "igraph", "leiden_n_iterations": 2, "seed": args.seed,
            "scanpy": sc.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"input": str(args.corrupted), "kcluster": result["kcluster"]}))

if __name__ == "__main__":
    main()
