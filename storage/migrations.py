from __future__ import annotations

from pathlib import Path

from legopolitics.constants import DATABASE_SCHEMA_VERSION


def inspect_schema(path: Path) -> str:
    import sqlite3

    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT value FROM schema_info WHERE key='schema_version'"
        ).fetchone()
    return str(row[0]) if row else "unknown"


def migrate_output(path: Path) -> str:
    version = inspect_schema(path)
    if version == DATABASE_SCHEMA_VERSION:
        return version
    raise ValueError(f"No migration registered from {version} to {DATABASE_SCHEMA_VERSION}")
