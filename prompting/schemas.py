from __future__ import annotations

from pydantic import BaseModel, Field


class ObservationOutput(BaseModel):
    visible_evidence: list[str] = Field(default_factory=list)
    uncertain_observations: list[str] = Field(default_factory=list)
    cannot_determine: list[str] = Field(default_factory=list)
    label: str | None = None
    confidence: float | None = None


class InterpretationOutput(BaseModel):
    interpretation: str
    direct_visual_evidence: list[str] = Field(default_factory=list)
    textual_evidence: list[str] = Field(default_factory=list)
    audio_evidence: list[str] = Field(default_factory=list)
    contextual_inference: list[str] = Field(default_factory=list)
    contradictory_evidence: list[str] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    confidence: float | None = None
