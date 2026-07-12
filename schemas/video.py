from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from legopolitics.schemas.common import SchemaModel


class VideoRecord(SchemaModel):
    video_id: str
    path: Path
    normalized_path: str
    fingerprint: str
    file_size: int
    modified_ns: int
    duration_seconds: float | None = None
    frame_count: int | None = None
    average_fps: float | None = None
    nominal_fps: float | None = None
    variable_frame_rate: bool | None = None
    width: int | None = None
    height: int | None = None
    rotation: int | None = None
    pixel_format: str | None = None
    codec: str | None = None
    bit_rate: int | None = None
    time_base: str | None = None
    has_audio: bool = False
    audio_streams: int = 0
    subtitle_streams: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneRecord(SchemaModel):
    scene_id: str
    video_id: str
    start_frame_index: int
    end_frame_index: int
    start_timestamp: float
    end_timestamp: float
    representative_frame_ids: list[str] = Field(default_factory=list)
    change_score: float | None = None
