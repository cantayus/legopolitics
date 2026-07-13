from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from legopolitics.schemas.audio import AudioEvent, DiarizationSegment, TranscriptSegment
from legopolitics.schemas.common import OutputPaths, SchemaModel, utc_now
from legopolitics.schemas.detection import DetectionRecord
from legopolitics.schemas.event import EventRecord
from legopolitics.schemas.frame import FrameRecord
from legopolitics.schemas.mask import MaskRecord
from legopolitics.schemas.narrative import CodebookScore, NarrativeSegment
from legopolitics.schemas.ocr import OCRRecord
from legopolitics.schemas.provenance import RunProvenance
from legopolitics.schemas.relation import RelationRecord
from legopolitics.schemas.track import TrackObservation, TrackRecord
from legopolitics.schemas.validation import ValidationRecord
from legopolitics.schemas.video import SceneRecord, VideoRecord
from legopolitics.schemas.visual import AttributeRecord, VisualResponse


class ErrorRecord(SchemaModel):
    error_id: str
    video_id: str | None = None
    stage: str
    item_id: str | None = None
    exception_type: str
    message: str
    traceback: str | None = None
    retry_count: int = 0
    recoverable: bool = True
    configuration_hash: str | None = None
    model_id: str | None = None
    device: str | None = None
    memory_state: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ModelAgreementRecord(SchemaModel):
    agreement_id: str
    video_id: str
    unit_type: str
    unit_id: str
    label: str | None = None
    model_count: int
    support_count: int
    majority_label: str | None = None
    maximum_confidence_label: str | None = None
    weighted_label: str | None = None
    mean_confidence: float | None = None
    median_confidence: float | None = None
    minimum_confidence: float | None = None
    maximum_confidence: float | None = None
    entropy: float | None = None
    pairwise_agreement: float | None = None
    disagreement_flag: bool = False
    human_review_priority: float | None = None


class AnalysisResult(SchemaModel):
    video: VideoRecord
    scenes: list[SceneRecord] = Field(default_factory=list)
    frames: list[FrameRecord] = Field(default_factory=list)
    detections: list[DetectionRecord] = Field(default_factory=list)
    fused_detections: list[DetectionRecord] = Field(default_factory=list)
    masks: list[MaskRecord] = Field(default_factory=list)
    tracks: list[TrackRecord] = Field(default_factory=list)
    track_observations: list[TrackObservation] = Field(default_factory=list)
    attributes: list[AttributeRecord] = Field(default_factory=list)
    visual_responses: list[VisualResponse] = Field(default_factory=list)
    ocr: list[OCRRecord] = Field(default_factory=list)
    transcript: list[TranscriptSegment] = Field(default_factory=list)
    diarization: list[DiarizationSegment] = Field(default_factory=list)
    audio_events: list[AudioEvent] = Field(default_factory=list)
    relations: list[RelationRecord] = Field(default_factory=list)
    events: list[EventRecord] = Field(default_factory=list)
    narrative_segments: list[NarrativeSegment] = Field(default_factory=list)
    codebook_scores: list[CodebookScore] = Field(default_factory=list)
    model_agreement: list[ModelAgreementRecord] = Field(default_factory=list)
    validation_summary: list[ValidationRecord] = Field(default_factory=list)
    errors: list[ErrorRecord] = Field(default_factory=list)
    provenance: RunProvenance | None = None
    output_paths: OutputPaths
