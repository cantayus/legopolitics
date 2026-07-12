from __future__ import annotations

from typing import Any

from pydantic import Field

from legopolitics.schemas.common import SchemaModel


class TranscriptSegment(SchemaModel):
    segment_id: str
    video_id: str
    start_timestamp: float
    end_timestamp: float
    original_text: str
    translated_text: str | None = None
    language: str | None = None
    speaker_label: str | None = None
    confidence: float | None = None
    words: list[dict[str, Any]] = Field(default_factory=list)
    backend: str
    model_revision: str | None = None


class DiarizationSegment(SchemaModel):
    diarization_id: str
    video_id: str
    start_timestamp: float
    end_timestamp: float
    speaker_label: str
    confidence: float | None = None
    backend: str


class AudioEvent(SchemaModel):
    audio_event_id: str
    video_id: str
    start_timestamp: float
    end_timestamp: float
    label: str
    confidence: float
    backend: str
    evidence_frame_ids: list[str] = Field(default_factory=list)
