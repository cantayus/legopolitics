from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def to_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=json_default, ensure_ascii=False))
