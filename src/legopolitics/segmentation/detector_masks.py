from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from legopolitics.schemas import DetectionRecord, MaskRecord, ModelProvenance
from legopolitics.segmentation.mask_utils import mask_to_polygons, save_mask, touches_border


class BoxMaskSegmenter:
    """Dependency-free fallback that converts detector boxes into rectangular masks."""

    def load(self) -> None:
        return

    def segment(
        self, image_path: Path, detections: Sequence[DetectionRecord], masks_dir: Path
    ) -> list[MaskRecord]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(image_path)
        height, width = image.shape[:2]
        records: list[MaskRecord] = []
        for detection in detections:
            mask = np.zeros((height, width), dtype=np.uint8)
            x1 = max(0, int(detection.box.x1))
            y1 = max(0, int(detection.box.y1))
            x2 = min(width, int(detection.box.x2))
            y2 = min(height, int(detection.box.y2))
            mask[y1:y2, x1:x2] = 1
            mask_id = f"mask_{uuid.uuid4().hex[:16]}"
            path = masks_dir / f"{mask_id}.png"
            save_mask(mask, path)
            records.append(
                MaskRecord(
                    mask_id=mask_id,
                    video_id=detection.video_id,
                    frame_id=detection.frame_id,
                    detection_id=detection.detection_id,
                    path=path,
                    polygon_json=mask_to_polygons(mask),
                    area_pixels=int(mask.sum()),
                    visible_fraction=1.0,
                    touches_border=touches_border(mask),
                    confidence=detection.confidence,
                )
            )
            detection.mask_id = mask_id
        return records

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="box_mask", model_id="detector-boxes")

    def unload(self) -> None:
        return
