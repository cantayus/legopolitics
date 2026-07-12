from __future__ import annotations

import json
from typing import Any


def render(template: str, **values: Any) -> str:
    prepared = {
        key: (
            json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in values.items()
    }
    return template.format(**prepared)
