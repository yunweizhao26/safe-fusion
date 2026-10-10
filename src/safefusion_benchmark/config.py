from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

from .hashing import canonical_json, sha256_bytes


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Study configuration must be a mapping")
    return value


def git_commit(root: str | Path = ".") -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def config_hash(config: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(config).encode())


def run_id(config: dict[str, Any], root: str | Path = ".") -> str:
    return f"{git_commit(root)[:12]}-{config_hash(config)[:12]}"


def enabled_names(config: dict[str, Any], section: str) -> list[str]:
    return [name for name, spec in config[section].items() if spec.get("enabled", True)]
