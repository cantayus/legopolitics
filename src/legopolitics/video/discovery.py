from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

from legopolitics.constants import SUPPORTED_VIDEO_EXTENSIONS
from legopolitics.utils.fingerprints import deterministic_partition, fast_file_fingerprint

LOGGER = logging.getLogger("legopolitics")


def normalize_path(path: Path) -> str:
    try:
        return str(path.resolve(strict=False))
    except OSError:
        return str(path.absolute())


def discover_videos(
    input_path: str | Path,
    recursive: bool = True,
    allowed_extensions: set[str] | None = None,
    scan_progress_every: int = 1000,
    num_batches: int = 1,
    batch_id: int = 0,
) -> list[Path]:
    source = Path(input_path)
    extensions = {e.lower() for e in (allowed_extensions or SUPPORTED_VIDEO_EXTENSIONS)}
    if not 0 <= batch_id < num_batches:
        raise ValueError("batch_id must be in [0, num_batches)")
    if source.is_file():
        candidates: Iterator[Path] = iter([source])
    elif source.is_dir():
        candidates = source.rglob("*") if recursive else source.glob("*")
    else:
        raise FileNotFoundError(source)

    found: list[Path] = []
    scanned = 0
    seen_fingerprints: set[str] = set()
    for path in candidates:
        scanned += 1
        if scan_progress_every > 0 and scanned % scan_progress_every == 0:
            LOGGER.info("Scanned %s filesystem entries; found %s videos", scanned, len(found))
        if not path.is_file() or path.name.startswith(".") or path.name.startswith("~"):
            continue
        if path.suffix.lower() not in extensions:
            continue
        try:
            if path.stat().st_size == 0:
                LOGGER.warning("Skipping zero-byte video: %s", path)
                continue
            fingerprint = fast_file_fingerprint(path)
        except OSError as exc:
            LOGGER.warning("Skipping unreadable path %s: %s", path, exc)
            continue
        if fingerprint in seen_fingerprints:
            LOGGER.info("Skipping duplicate video: %s", path)
            continue
        seen_fingerprints.add(fingerprint)
        identity = f"{normalize_path(path)}::{fingerprint}"
        if deterministic_partition(identity, num_batches) != batch_id:
            continue
        found.append(path)
    return sorted(found, key=lambda p: normalize_path(p).casefold())
