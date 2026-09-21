from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import sha256_bytes, sha256_file


REQUIRED_METADATA = {
    "method", "scale", "cell_ids", "gene_ids", "gene_order_sha256",
    "cell_order_sha256", "training_splits", "parameters", "seed",
    "input_sha256", "coordinates_sha256", "method_version",
}


def order_hash(values: list[str]) -> str:
    return sha256_bytes("\n".join(values).encode())


def write_output_contract(
    directory: str | Path,
    mean: np.ndarray,
    metadata: dict[str, Any],
    variance: np.ndarray | None = None,
    fill_score: np.ndarray | None = None,
    embedding: np.ndarray | None = None,
) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "mean.npy", np.asarray(mean, dtype=np.float32), allow_pickle=False)
    for name, value in (("variance", variance), ("fill_score", fill_score), ("embedding", embedding)):
        if value is not None:
            np.save(destination / f"{name}.npy", np.asarray(value, dtype=np.float32), allow_pickle=False)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_output_contract(directory: str | Path, cell_ids: list[str], gene_ids: list[str]) -> dict[str, Any]:
    source = Path(directory)
    mean_path = source / "mean.npy"
    metadata_path = source / "metadata.json"
    if not mean_path.exists() or not metadata_path.exists():
        raise ValueError("method output must contain mean.npy and metadata.json")
    mean = np.load(mean_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    missing = REQUIRED_METADATA - set(metadata)
    if missing:
        raise ValueError(f"missing provenance fields: {sorted(missing)}")
    if mean.dtype != np.float32:
        raise ValueError(f"mean must be float32, got {mean.dtype}")
    if mean.shape != (len(cell_ids), len(gene_ids)):
        raise ValueError(f"mean shape {mean.shape} does not match {(len(cell_ids), len(gene_ids))}")
    if metadata["cell_ids"] != cell_ids or metadata["gene_ids"] != gene_ids:
        raise ValueError("cell/gene order mismatch")
    if metadata["cell_order_sha256"] != order_hash(cell_ids) or metadata["gene_order_sha256"] != order_hash(gene_ids):
        raise ValueError("cell/gene order hash mismatch")
    if not np.isfinite(mean).all():
        raise ValueError("mean contains NaN or infinite values")
    if metadata["scale"] == "counts" and np.any(mean < 0):
        raise ValueError("count-scale mean contains negative values")
    if "test" in metadata["training_splits"]:
        raise ValueError("held-out test values entered training")
    optional = {}
    for name in ("variance", "fill_score"):
        path = source / f"{name}.npy"
        if path.exists():
            value = np.load(path, allow_pickle=False)
            if value.shape != mean.shape or not np.isfinite(value).all():
                raise ValueError(f"invalid optional output: {name}")
            optional[name] = sha256_file(path)
    return {"valid": True, "mean_sha256": sha256_file(mean_path), "optional": optional}
