from __future__ import annotations

from typing import Any

from pydantic import Field

from legopolitics.schemas.common import Evidence, SchemaModel


class NarrativeSegment(SchemaModel):
    narrative_segment_id: str
    video_id: str
    label: str
    start_timestamp: float
    end_timestamp: float
    scene_ids: list[str] = Field(default_factory=list)
    track_ids: list[str] = Field(default_factory=list)
    relation_ids: list[str] = Field(default_factory=list)
    dominant_actions: list[str] = Field(default_factory=list)
    confidence: float
    evidence: Evidence = Field(default_factory=Evidence)


class CodebookScore(SchemaModel):
    score_id: str
    video_id: str
    unit_type: str
    unit_id: str
    variable_id: str
    applicable: bool = True
    score: float | str | bool | None = None
    scale_min: float | None = None
    scale_max: float | None = None
    confidence: float | None = None
    evidence: Evidence = Field(default_factory=Evidence)
    model_id: str | None = None
    prompt_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
