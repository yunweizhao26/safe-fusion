#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safefusion_benchmark.contracts import order_hash, write_output_contract
from safefusion_benchmark.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrupted", required=True)
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gene-col", default="feature_name")
    parser.add_argument("--max-length", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gene-batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset, SequentialSampler

    from scgpt.data_collator import DataCollator
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    from scgpt.utils import load_pretrained

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    source = ad.read_h5ad(args.corrupted)
    matrix = source.layers["corrupted_counts"]
    counts = matrix.toarray().astype(np.float32) if sparse.issparse(matrix) else np.asarray(matrix, dtype=np.float32)
    gene_names = (
        source.var[args.gene_col].astype(str).to_numpy()
        if args.gene_col in source.var
        else source.var_names.astype(str).to_numpy()
    )

    checkpoint = Path(args.ckpt_dir)
    vocab = GeneVocab.from_file(checkpoint / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab["<pad>"])
    with (checkpoint / "args.json").open() as handle:
        config = json.load(handle)

    matched = np.asarray([name in vocab for name in gene_names], dtype=bool)
    matched_indices = np.flatnonzero(matched)
    matched_vocab_ids = np.asarray([vocab[name] for name in gene_names[matched]], dtype=np.int64)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config["embsize"],
        nhead=config["nheads"],
        d_hid=config["d_hid"],
        nlayers=config["nlayers"],
        nlayers_cls=config["n_layers_cls"],
        n_cls=1,
        vocab=vocab,
        dropout=config["dropout"],
        pad_token=config["pad_token"],
        pad_value=config["pad_value"],
        do_mvc=True,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )
    load_pretrained(
        model,
        torch.load(checkpoint / "best_model.pt", map_location=device),
        verbose=False,
    )
    model.to(device)
    model.eval()

    class ObservedGeneDataset(Dataset):
        def __len__(self) -> int:
            return len(counts)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
            row = counts[index, matched_indices]
            expressed = np.flatnonzero(row > 0)
            genes = matched_vocab_ids[expressed]
            values = row[expressed]
            genes = np.insert(genes, 0, vocab["<cls>"])
            values = np.insert(values, 0, config["pad_value"])
            return {
                "id": index,
                "genes": torch.from_numpy(genes).long(),
                "expressions": torch.from_numpy(values).float(),
            }

    dataset = ObservedGeneDataset()
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[config["pad_token"]],
        pad_value=config["pad_value"],
        do_mlm=False,
        do_binning=True,
        max_length=args.max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    cell_embeddings = np.zeros((len(dataset), config["embsize"]), dtype=np.float32)
    offset = 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        for batch in loader:
            genes = batch["gene"].to(device)
            values = batch["expr"].to(device)
            encoded = model._encode(
                genes,
                values,
                src_key_padding_mask=genes.eq(vocab[config["pad_token"]]),
                batch_labels=None,
            )
            embedding = encoded[:, 0, :].float().cpu().numpy()
            cell_embeddings[offset : offset + len(embedding)] = embedding
            offset += len(embedding)

    scores = np.full(counts.shape, np.nan, dtype=np.float32)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        for cell_start in range(0, len(counts), args.batch_size):
            cell_stop = min(cell_start + args.batch_size, len(counts))
            cell_embedding = torch.from_numpy(cell_embeddings[cell_start:cell_stop]).to(device)
            for gene_start in range(0, len(matched_indices), args.gene_batch_size):
                gene_stop = min(gene_start + args.gene_batch_size, len(matched_indices))
                token_ids = torch.from_numpy(matched_vocab_ids[gene_start:gene_stop]).to(device)
                gene_embedding = model.encoder(token_ids.unsqueeze(0)).expand(
                    len(cell_embedding), -1, -1
                )
                prediction = model.mvc_decoder(cell_embedding, gene_embedding)["pred"]
                scores[
                    cell_start:cell_stop,
                    matched_indices[gene_start:gene_stop],
                ] = prediction.float().cpu().numpy()

    unmatched_score = float(np.nanmin(scores) - 1.0)
    scores[~np.isfinite(scores)] = unmatched_score
    try:
        version = subprocess.check_output(
            ["git", "-C", str(ROOT / "baselines_and_data" / "scGPT"), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        version = "scgpt-0.2.4"

    cell_ids = source.obs_names.astype(str).tolist()
    gene_ids = source.var_names.astype(str).tolist()
    metadata = {
        "method": "scgpt_mvc",
        "method_version": version,
        "scale": "frozen_scgpt_masked_value_score",
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "cell_order_sha256": order_hash(cell_ids),
        "gene_order_sha256": order_hash(gene_ids),
        "training_splits": ["external_pretraining_only"],
        "parameters": {
            "frozen_checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint / "best_model.pt"),
            "decoder": "MVC masked value decoder",
            "test_used_for_fit": False,
            "matched_genes": int(matched.sum()),
            "total_genes": int(len(matched)),
            "max_observed_sequence_length": args.max_length,
            "unmatched_gene_score": unmatched_score,
            "disclosure": "Frozen zero shot masked value scores. No target dataset labels, hidden values, or model fitting are used. Genes absent from the pretrained vocabulary are ranked last.",
        },
        "seed": args.seed,
        "input_sha256": sha256_file(args.corrupted),
        "coordinates_sha256": sha256_file(args.coordinates),
    }
    write_output_contract(args.output, scores, metadata, embedding=cell_embeddings)
    print(json.dumps({"method": "scgpt_mvc", "shape": list(scores.shape), "parameters": metadata["parameters"]}))


if __name__ == "__main__":
    main()
