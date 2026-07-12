from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from legopolitics.schemas.common import SchemaModel


class TrackObservation(SchemaModel):
    track_id: str
    frame_id: str
    detection_id: str
    timestamp_seconds: float
    confidence: float
    screen_area_fraction: float | None = None
    centrality: float | None = None


class TrackRecord(SchemaModel):
    track_id: str
    video_id: str
    object_type: str
    first_frame_id: str
    last_frame_id: str
    first_timestamp: float
    last_timestamp: float
    observed_frame_count: int
    estimated_duration: float
    visible_duration: float
    screen_time: float
    mean_confidence: float
    minimum_confidence: float
    maximum_confidence: float
    mean_screen_area: float | None = None
    maximum_screen_area: float | None = None
    mean_centrality: float | None = None
    representative_crop_path: Path | None = None
    stable_attributes: dict[str, Any] = Field(default_factory=dict)
    group_assignment: str | None = None
    group_confidence: float | None = None
    role_assignments: list[str] = Field(default_factory=list)
    role_confidences: dict[str, float] = Field(default_factory=dict)
    track_quality_score: float | None = None
    possible_id_switch: bool = False
