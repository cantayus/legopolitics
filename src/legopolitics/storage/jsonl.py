from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from legopolitics.storage.atomic import atomic_write_text
from legopolitics.utils.serialization import json_default


def write_jsonl(path: Path, records: Iterable[Any]) -> Path:
    lines = []
    for record in records:
        if hasattr(record, "model_dump"):
            record = record.model_dump(mode="json")
        lines.append(json.dumps(record, ensure_ascii=False, default=json_default))
    return atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def append_jsonl(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(record, "model_dump"):
        record = record.model_dump(mode="json")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=json_default) + "\n")
