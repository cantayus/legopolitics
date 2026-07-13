from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fast_file_fingerprint(path: str | Path, sample_size: int = 1024 * 1024) -> str:
    source = Path(path)
    stat = source.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    with source.open("rb") as handle:
        digest.update(handle.read(sample_size))
        if stat.st_size > sample_size:
            handle.seek(max(0, stat.st_size - sample_size))
            digest.update(handle.read(sample_size))
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deterministic_partition(identity: str, num_batches: int) -> int:
    if num_batches <= 0:
        raise ValueError("num_batches must be greater than zero")
    value = int(hashlib.sha256(identity.encode("utf-8")).hexdigest(), 16)
    return value % num_batches
