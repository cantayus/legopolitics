from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legopolitics.storage.atomic import atomic_write_text


def save_stage_cache(path: Path, cache_key: str, payload: Any) -> Path:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    document = {"cache_key": cache_key, "payload": payload}
    return atomic_write_text(path, json.dumps(document, ensure_ascii=False, default=str))


def load_stage_cache(path: Path, cache_key: str) -> Any | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document.get("payload") if document.get("cache_key") == cache_key else None
