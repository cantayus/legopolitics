from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from legopolitics.detection.base import ImageInput
from legopolitics.schemas import BoundingBox, DetectionBatch, DetectionRecord, ModelProvenance


class MockDetector:
    """Deterministic detector used in tests and dependency-free demonstrations."""

    def __init__(self, class_name: str = "brick_figure") -> None:
        self.class_name = class_name
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def _shape(self, image: ImageInput) -> tuple[int, int]:
        if isinstance(image, np.ndarray):
            return image.shape[1], image.shape[0]
        loaded = cv2.imread(str(Path(image)))
        if loaded is None:
            raise OSError(f"Could not read image: {image}")
        return loaded.shape[1], loaded.shape[0]

    def predict(
        self, images: Sequence[ImageInput], frame_ids: Sequence[str], video_id: str
    ) -> list[DetectionBatch]:
        if not self.loaded:
            self.load()
        batches: list[DetectionBatch] = []
        for image, frame_id in zip(images, frame_ids, strict=True):
            width, height = self._shape(image)
            box = BoundingBox(x1=width * 0.25, y1=height * 0.15, x2=width * 0.75, y2=height * 0.90)
            record = DetectionRecord(
                detection_id=f"det_mock_{frame_id}",
                video_id=video_id,
                frame_id=frame_id,
                detector_id="mock",
                class_id=0,
                class_name=self.class_name,
                confidence=0.50,
                box=box,
                normalized_box=BoundingBox(x1=0.25, y1=0.15, x2=0.75, y2=0.90),
                frame_area_fraction=box.area / (width * height),
                model_id="mock-detector",
            )
            batches.append(DetectionBatch(frame_id=frame_id, detections=[record]))
        return batches

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="mock", model_id="mock-detector")

    def unload(self) -> None:
        self.loaded = False
