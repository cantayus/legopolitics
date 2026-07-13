from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except Exception:
        with suppress(OSError):
            os.unlink(temp)
        raise
    return path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(path, text.encode(encoding))


def atomic_build(path: Path, builder: Callable[[Path], T]) -> T:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp" + path.suffix)
    try:
        result = builder(temp)
        os.replace(temp, path)
        return result
    finally:
        if temp.exists():
            temp.unlink()
