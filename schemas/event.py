from __future__ import annotations

from pydantic import Field

from legopolitics.schemas.common import Evidence, SchemaModel


class EventRecord(SchemaModel):
    event_id: str
    video_id: str
    label: str
    start_timestamp: float
    end_timestamp: float
    participating_track_ids: list[str] = Field(default_factory=list)
    detection_ids: list[str] = Field(default_factory=list)
    confidence: float
    evidence: Evidence = Field(default_factory=Evidence)
    alternative_labels: list[str] = Field(default_factory=list)
