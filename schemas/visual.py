from __future__ import annotations

from typing import Any

from pydantic import Field

from legopolitics.schemas.common import Evidence, ModelProvenance, SchemaModel


class VisualResponse(SchemaModel):
    response_id: str
    target_type: str
    target_id: str
    pass_type: str
    prompt_version: str
    raw_text: str
    parsed: dict[str, Any] | None = None
    confidence: float | None = None
    evidence: Evidence = Field(default_factory=Evidence)
    provenance: ModelProvenance


class AttributeRecord(SchemaModel):
    attribute_id: str
    video_id: str
    frame_id: str | None = None
    track_id: str | None = None
    detection_id: str | None = None
    name: str
    value: Any
    confidence: float | None = None
    evidence: Evidence = Field(default_factory=Evidence)
    model_id: str | None = None
    prompt_version: str | None = None
