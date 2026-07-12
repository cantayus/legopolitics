from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd


def flatten_record(record: Any) -> dict[str, Any]:
    if hasattr(record, "model_dump"):
        data = record.model_dump(mode="json")
    elif isinstance(record, dict):
        data = dict(record)
    else:
        data = vars(record)
    flat = {}
    for key, value in data.items():
        flat[key] = (
            json.dumps(value, ensure_ascii=False, default=str)
            if isinstance(value, (dict, list))
            else value
        )
    return flat


def records_frame(records: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([flatten_record(record) for record in records])


def write_table(database: Path, table: str, records: list[Any]) -> None:
    frame = records_frame(records)
    if frame.empty:
        frame = pd.DataFrame({"_empty": pd.Series(dtype="string")})
    connection = sqlite3.connect(database)
    try:
        frame.to_sql(table, connection, if_exists="replace", index=False)
        connection.commit()
    finally:
        connection.close()
