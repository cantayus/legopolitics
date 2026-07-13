from __future__ import annotations

from pydantic import Field

from legopolitics.schemas.common import SchemaModel
from legopolitics.schemas.detection import BoundingBox


class OCRRecord(SchemaModel):
    text_region_id: str
    video_id: str
    frame_id: str
    track_id: str | None = None
    bounding_box: BoundingBox | None = None
    original_text: str
    normalized_text: str
    detected_language: str | None = None
    translated_text: str | None = None
    confidence: float | None = None
    backend: str
    text_type: str | None = None
    first_appearance: float | None = None
    last_appearance: float | None = None
    repetition_count: int = 1
    temporal_cluster_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
