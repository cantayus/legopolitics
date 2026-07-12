from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from legopolitics.config.models import AnalysisConfig


def load_config(
    path: str | Path | None = None, env_prefix: str = "LEGOPOLITICS__"
) -> AnalysisConfig:
    config = AnalysisConfig() if path is None else AnalysisConfig.from_yaml(path)
    overrides: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(env_prefix):
            continue
        dotted = key[len(env_prefix) :].lower().replace("__", ".")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        overrides[dotted] = value
    return config.with_overrides(**overrides) if overrides else config
