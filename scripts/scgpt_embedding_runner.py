#!/usr/bin/env python3
"""Extract frozen scGPT cell embeddings for a dataset (official embed_data)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scgpt.tasks import embed_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gene-col", default="feature_name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    result = embed_data(
        adata_or_file=args.input,
        model_dir=args.ckpt_dir,
        gene_col=args.gene_col,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
        obs_to_save=None,
        return_new_adata=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    embeddings = result.X.toarray() if hasattr(result.X, "toarray") else np.asarray(result.X)
    np.save(output, embeddings, allow_pickle=False)
    print("saved", output, embeddings.shape)


if __name__ == "__main__":
    main()
