from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from legopolitics.schemas.common import SchemaModel


class BoundingBox(SchemaModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class DetectionRecord(SchemaModel):
    detection_id: str
    video_id: str
    frame_id: str
    detector_id: str
    class_id: int | None = None
    class_name: str
    confidence: float
    box: BoundingBox
    normalized_box: BoundingBox | None = None
    frame_area_fraction: float | None = None
    mask_id: str | None = None
    keypoints: list[list[float]] | None = None
    model_id: str | None = None
    model_revision: str | None = None
    weight_hash: str | None = None
    source_detection_ids: list[str] = Field(default_factory=list)
    crop_path: Path | None = None
    context_crop_path: Path | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class DetectionBatch(SchemaModel):
    frame_id: str
    detections: list[DetectionRecord] = Field(default_factory=list)
