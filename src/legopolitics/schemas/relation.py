from __future__ import annotations

from pydantic import Field

from legopolitics.schemas.common import Evidence, SchemaModel


class RelationRecord(SchemaModel):
    relation_id: str
    video_id: str
    scene_id: str | None = None
    start_timestamp: float
    end_timestamp: float
    subject_track_id: str
    predicate: str
    object_track_id: str | None = None
    confidence: float
    evidence: Evidence = Field(default_factory=Evidence)
    alternative_predicates: list[str] = Field(default_factory=list)
