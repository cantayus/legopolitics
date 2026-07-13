from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(SchemaModel):
    direct_visual: list[str] = Field(default_factory=list)
    textual: list[str] = Field(default_factory=list)
    audio: list[str] = Field(default_factory=list)
    contextual: list[str] = Field(default_factory=list)
    contradictory: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    frame_ids: list[str] = Field(default_factory=list)
    track_ids: list[str] = Field(default_factory=list)
    ocr_ids: list[str] = Field(default_factory=list)
    audio_ids: list[str] = Field(default_factory=list)


class ModelProvenance(SchemaModel):
    adapter: str
    model_id: str | None = None
    revision: str | None = None
    weight_hash: str | None = None
    device: str | None = None
    dtype: str | None = None
    is_remote: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutputPaths(SchemaModel):
    root: Path
    workbook: Path | None = None
    report: Path | None = None
    manifest: Path | None = None
    tables_dir: Path | None = None
    jsonl_dir: Path | None = None
    frames_dir: Path | None = None
    crops_dir: Path | None = None
    masks_dir: Path | None = None
    audio_dir: Path | None = None
