from __future__ import annotations

from pathlib import Path

from pydantic import Field

from legopolitics.schemas.common import SchemaModel


class FrameQuality(SchemaModel):
    blur_score: float | None = None
    brightness: float | None = None
    contrast: float | None = None
    saturation: float | None = None
    black_frame_score: float | None = None
    underexposure_score: float | None = None
    overexposure_score: float | None = None
    compression_artifact_score: float | None = None
    occlusion_estimate: float | None = None
    readability_score: float | None = None
    crop_suitability_score: float | None = None
    is_duplicate: bool = False
    is_corrupt: bool = False


class FrameRecord(SchemaModel):
    video_id: str
    frame_id: str
    source_frame_index: int
    timestamp_seconds: float
    timestamp_ms: int
    source_fps: float
    scene_id: str | None = None
    shot_id: str | None = None
    sampling_reasons: list[str] = Field(default_factory=list)
    sampling_priority: float = 0.0
    previous_sample_timestamp: float | None = None
    next_sample_timestamp: float | None = None
    perceptual_hash: str | None = None
    motion_score: float | None = None
    scene_change_score: float | None = None
    image_path: Path | None = None
    decode_status: str = "ok"
    width: int | None = None
    height: int | None = None
    quality: FrameQuality = Field(default_factory=FrameQuality)
