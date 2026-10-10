#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file

HERE = Path(__file__).resolve().parent
METHODS = {
    "dca": {"command": [str(ROOT / ".conda-dca/bin/python"), str(HERE / "run_dca.py")], "kcluster": False,
            "scale": "counts", "probability": False, "extra": ["nb-conddisp"]},
    "dca_zinb": {"command": [str(ROOT / ".conda-dca/bin/python"), str(HERE / "run_dca.py")], "kcluster": False,
                 "scale": "counts", "probability": True, "extra": ["zinb-conddisp"]},
    "scimpute": {"command": [str(ROOT / ".conda-scimpute/bin/Rscript"), str(HERE / "run_scimpute.R")], "kcluster": True,
                 "scale": "counts", "probability": True},
    "screcover": {"command": [str(ROOT / ".conda-screcover/bin/Rscript"), str(HERE / "run_screcover.R")], "kcluster": True,
                  "scale": "counts", "probability": True},
    "enimpute": {"command": [str(ROOT / ".conda-enimpute/bin/Rscript"), str(HERE / "run_enimpute.R")], "kcluster": True,
                 "scale": "normalized_expression_1e4", "probability": False,

                 "environment": {"PATH": str(ROOT / ".conda-enimpute/bin"), "RETICULATE_PYTHON": str(ROOT / ".conda-enimpute/bin/python")}},
}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--corrupted", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kcluster-json", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None, help="Defaults to <output>/work; bulk files are removed after reading.")
    parser.add_argument("--ncores", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    spec = METHODS[args.method]
    if (args.output / "mean.npy").exists():
        raise FileExistsError(f"{args.output} already holds a contract")
    source = ad.read_h5ad(args.corrupted)
    matrix = source.layers["corrupted_counts"]
    counts = matrix.tocsr() if sparse.issparse(matrix) else sparse.csr_matrix(np.asarray(matrix))
    if counts.data.size and not np.all(np.equal(np.mod(counts.data, 1), 0)):
        raise ValueError("the input holds non-integer counts")
    n_cells, n_genes = counts.shape
    cell_ids = source.obs_names.astype(str).tolist()
    gene_ids = source.var_names.astype(str).tolist()
    split = pd.read_parquet(args.splits).set_index("cell_id").loc[cell_ids, "split"].to_numpy()

    work = (args.work_dir or args.output / "work").resolve()
    work.mkdir(parents=True, exist_ok=True)
    io.mmwrite(str(work / "counts.mtx"), sparse.csr_matrix(counts.T, dtype=np.int64), field="integer")
    (work / "genes.txt").write_text("\n".join(gene_ids) + "\n")
    (work / "cells.txt").write_text("\n".join(cell_ids) + "\n")

    command = [*spec["command"], str(work)]
    kcluster = None
    if spec["kcluster"]:
        if args.kcluster_json is None:
            parser.error(f"--kcluster-json is required for {args.method}")
        kcluster = json.loads(args.kcluster_json.read_text())
        command.append(str(kcluster["kcluster"]))
    command += [str(args.ncores), str(args.seed), *spec.get("extra", [])]
    started = time.perf_counter()
    environment = dict(os.environ)

    with socket.socket() as probe:
        probe.bind(("", 0))
        environment["R_PARALLEL_PORT"] = str(probe.getsockname()[1])
    for name, value in spec.get("environment", {}).items():
        environment[name] = f"{value}:{environment.get('PATH', '')}" if name == "PATH" else value
    subprocess.run(command, check=True, cwd=work, env=environment)
    elapsed = time.perf_counter() - started

    def read(name: str) -> np.ndarray:
        values = np.fromfile(work / name, dtype=np.float64)
        if values.size != n_cells * n_genes:
            raise ValueError(f"{name} holds {values.size} values for a {n_cells} x {n_genes} input")
        return values.reshape(n_cells, n_genes)

    mean = read("imputed.bin")
    if not np.isfinite(mean).all():
        raise ValueError(f"{args.method} returned non-finite values")
    mean = np.clip(mean, 0.0, None).astype(np.float32)
    runner = json.loads((work / "runner.json").read_text())
    metadata = {
        "method": args.method,
        "method_version": f"{runner.get('method')} {runner.get('package_version')} ({runner.get('source')})",
        "scale": spec["scale"],
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["transductive_full_corrupted_matrix"],
        "parameters": {
            "standard_usage": True,
            "transductive": True,
            "test_used_for_fit": True,
            "fit_cells": int(n_cells),
            "heldout_test_cells": int(np.sum(split == "test")),
            "kcluster": kcluster,
            "runner": runner,
            "ncores": args.ncores,
            "elapsed_seconds": elapsed,
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.corrupted),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, mean, metadata)
    if spec["probability"]:
        probability = read("dropout.bin")
        if not np.isfinite(probability).all():
            raise ValueError(f"{args.method} returned a non-finite dropout probability")
        np.save(args.output / "dropout_probability.npy", probability.astype(np.float32), allow_pickle=False)
    for name in ("counts.mtx", "counts.rds", "imputed.bin", "dropout.bin"):
        (work / name).unlink(missing_ok=True)
    for name in ("scimpute", "screcover", "lnorm"):
        shutil.rmtree(work / name, ignore_errors=True)
    zeros = counts.toarray() == 0
    print(json.dumps({
        "method": args.method, "output": str(args.output), "shape": [n_cells, n_genes], "kcluster": kcluster and kcluster["kcluster"],
        "share_of_zeros_with_positive_value": float(np.mean(mean[zeros] > 0)), "elapsed_seconds": round(elapsed, 1),
    }))

if __name__ == "__main__":
    main()
