from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from legopolitics.exceptions import DependencyUnavailableError
from legopolitics.storage.sqlite import records_frame


def parquet_frame(records: list[Any]) -> pd.DataFrame:
    """Return an Arrow-compatible frame without losing heterogeneous values.

    Research tables such as ``Attributes`` intentionally allow a value column to
    contain numbers, strings, booleans, or structured JSON. Arrow requires a
    single type per column, so mixed object columns are serialized as JSON text.
    Homogeneous numeric and boolean columns remain typed.
    """
    frame = records_frame(records)
    if frame.empty:
        return pd.DataFrame({"_empty": pd.Series(dtype="string")})

    for column in frame.columns:
        series = frame[column]
        if series.dtype != object:
            continue
        values = [value for value in series.tolist() if value is not None]
        value_types = {type(value) for value in values}
        if value_types <= {str}:
            frame[column] = series.astype("string")
            continue
        if value_types <= {bool}:
            frame[column] = series.astype("boolean")
            continue
        if value_types <= {int}:
            frame[column] = pd.to_numeric(series, errors="coerce").astype("Int64")
            continue
        if value_types <= {int, float}:
            frame[column] = pd.to_numeric(series, errors="coerce")
            continue
        frame[column] = series.map(
            lambda value: (
                None
                if value is None
                else value
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, default=str)
            )
        ).astype("string")
    return frame


def write_parquet(path: Path, records: list[Any], strict: bool = False) -> Path | None:
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        if strict:
            raise DependencyUnavailableError(
                "Parquet export requires legopolitics[parquet]"
            ) from exc
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet_frame(records).to_parquet(path, index=False)
    return path
