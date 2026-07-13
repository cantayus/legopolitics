from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import DetectionRecord, MaskRecord, ModelProvenance
from legopolitics.segmentation.mask_utils import mask_to_polygons, save_mask, touches_border


class Sam2Segmenter:
    """SAM 2 image segmenter using the Transformers SAM2 interface when installed."""

    def __init__(self, model_id: str = "facebook/sam2-hiera-small", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        try:
            import torch
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise DependencyUnavailableError(
                "SAM 2 support requires a Transformers release containing Sam2Model and Sam2Processor"
            ) from exc
        resolved = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
        if resolved == "auto":
            resolved = "cpu"
        self.processor = Sam2Processor.from_pretrained(self.model_id)
        self.model = Sam2Model.from_pretrained(self.model_id).to(resolved)
        self.device = resolved

    def segment(
        self, image_path: Path, detections: Sequence[DetectionRecord], masks_dir: Path
    ) -> list[MaskRecord]:
        if self.model is None or self.processor is None:
            self.load()
        import torch

        image = Image.open(image_path).convert("RGB")
        boxes = [[[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections]]
        if not detections:
            return []
        try:
            inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )[0]
        except Exception as exc:
            raise ModelInferenceError(f"SAM 2 inference failed: {exc}") from exc
        records: list[MaskRecord] = []
        for detection, mask_tensor in zip(detections, masks, strict=False):
            mask = np.asarray(mask_tensor.squeeze() > 0, dtype=np.uint8)
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
        return ModelProvenance(adapter="sam2", model_id=self.model_id, device=self.device)

    def unload(self) -> None:
        self.processor = None
        self.model = None
