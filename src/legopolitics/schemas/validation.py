from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from legopolitics.schemas.common import SchemaModel, utc_now


class ValidationRecord(SchemaModel):
    validation_id: str
    video_id: str
    unit_type: str
    unit_id: str
    sampling_reason: str
    suggested_label: str | None = None
    suggested_confidence: float | None = None
    human_label: str | None = None
    accepted: bool | None = None
    ambiguous: bool = False
    coder_id: str | None = None
    notes: str | None = None
    adjudication_status: str = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ManualCorrection(SchemaModel):
    correction_id: str
    video_id: str
    unit_type: str
    unit_id: str
    action: str
    original_value: Any = None
    corrected_value: Any = None
    coder_id: str | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
