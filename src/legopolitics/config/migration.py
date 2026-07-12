from __future__ import annotations

from typing import Any

from legopolitics.constants import CONFIG_SCHEMA_VERSION
from legopolitics.exceptions import ConfigurationError


def migrate_config(payload: dict[str, Any]) -> dict[str, Any]:
    version = str(payload.get("schema_version", CONFIG_SCHEMA_VERSION))
    if version == CONFIG_SCHEMA_VERSION:
        return payload
    raise ConfigurationError(f"No configuration migration is registered for schema {version}")
