from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from .hashing import hash_path, sha256_file


def _gpu_peak_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        values = [float(item.strip()) for item in output.splitlines() if item.strip()]
        return max(values) if values else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def environment_lock(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    lock = Path(path)
    return {"path": str(lock), "sha256": sha256_file(lock)} if lock.exists() else {"path": str(lock), "missing": True}


def execute_recorded(
    rule: str,
    provenance_path: str | Path,
    inputs: list[str | Path],
    outputs: list[str | Path],
    action: Callable[[], Any],
    *,
    seed: int | None = None,
    environment: str | Path | None = None,
    method: dict[str, Any] | None = None,
    suppress_failure: bool = False,
) -> Any:
    start = time.time()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    failure = None
    try:
        result = action()
        return result
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        if not suppress_failure:
            raise
        return None
    finally:
        usage_end = resource.getrusage(resource.RUSAGE_SELF)
        payload = {
            "schema_version": 1,
            "rule": rule,
            "command": list(sys.argv),
            "cwd": os.getcwd(),
            "started_at_unix": start,
            "finished_at_unix": time.time(),
            "runtime_seconds": time.time() - start,
            "peak_cpu_memory_mb": float(usage_end.ru_maxrss) / 1024.0,
            "user_cpu_seconds": usage_end.ru_utime - usage_start.ru_utime,
            "system_cpu_seconds": usage_end.ru_stime - usage_start.ru_stime,
            "peak_gpu_memory_mb": _gpu_peak_mb(),
            "seed": seed,
            "environment_lock": environment_lock(environment),
            "method": method,
            "inputs": [hash_path(path) for path in inputs],
            "outputs": [hash_path(path) for path in outputs],
            "failure": failure,
        }
        destination = Path(provenance_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(destination)
