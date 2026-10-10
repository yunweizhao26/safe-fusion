from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_path(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.exists():
        return {"path": str(target), "exists": False}
    if target.is_file():
        return {"path": str(target), "exists": True, "sha256": sha256_file(target), "bytes": target.stat().st_size}
    files = []
    for child in sorted(p for p in target.rglob("*") if p.is_file()):
        files.append({"relative_path": str(child.relative_to(target)), "sha256": sha256_file(child), "bytes": child.stat().st_size})
    return {"path": str(target), "exists": True, "files": files, "sha256": sha256_bytes(canonical_json(files).encode())}
