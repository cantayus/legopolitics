from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from legopolitics.detection.base import ImageInput
from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import BoundingBox, DetectionBatch, DetectionRecord, ModelProvenance


class GroundingDinoDetector:
    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        text_queries: list[str] | None = None,
        box_threshold: float = 0.30,
        device: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.text_queries = text_queries or ["brick-built figure"]
        self.box_threshold = box_threshold
        self.device = device
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Grounding DINO requires legopolitics[grounding]"
            ) from exc
        device = 0 if self.device == "auto" else self.device
        self.pipeline = pipeline("zero-shot-object-detection", model=self.model_id, device=device)

    @staticmethod
    def _pil(image: ImageInput) -> Image.Image:
        if isinstance(image, np.ndarray):
            return Image.fromarray(image[:, :, ::-1])
        return Image.open(Path(image)).convert("RGB")

    def predict(
        self, images: Sequence[ImageInput], frame_ids: Sequence[str], video_id: str
    ) -> list[DetectionBatch]:
        if self.pipeline is None:
            self.load()
        output: list[DetectionBatch] = []
        for image, frame_id in zip(images, frame_ids, strict=True):
            pil = self._pil(image)
            try:
                predictions = self.pipeline(
                    pil, candidate_labels=self.text_queries, threshold=self.box_threshold
                )
            except Exception as exc:
                raise ModelInferenceError(f"Grounding DINO inference failed: {exc}") from exc
            detections: list[DetectionRecord] = []
            for pred in predictions:
                box_data = pred["box"]
                box = BoundingBox(
                    x1=float(box_data["xmin"]),
                    y1=float(box_data["ymin"]),
                    x2=float(box_data["xmax"]),
                    y2=float(box_data["ymax"]),
                )
                detections.append(
                    DetectionRecord(
                        detection_id=f"det_{uuid.uuid4().hex[:16]}",
                        video_id=video_id,
                        frame_id=frame_id,
                        detector_id="grounding_dino",
                        class_name=str(pred["label"]),
                        confidence=float(pred["score"]),
                        box=box,
                        normalized_box=BoundingBox(
                            x1=box.x1 / pil.width,
                            y1=box.y1 / pil.height,
                            x2=box.x2 / pil.width,
                            y2=box.y2 / pil.height,
                        ),
                        frame_area_fraction=box.area / float(pil.width * pil.height),
                        model_id=self.model_id,
                    )
                )
            output.append(DetectionBatch(frame_id=frame_id, detections=detections))
        return output

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="grounding_dino", model_id=self.model_id, device=self.device)

    def unload(self) -> None:
        self.pipeline = None
