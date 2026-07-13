from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import OCRRecord
from legopolitics.utils.device import detect_device
from legopolitics.vision.common import to_pil


class TrOCRAdapter:
    def __init__(
        self, model_id: str = "microsoft/trocr-large-printed", device: str = "auto"
    ) -> None:
        self.model_id = model_id
        self.device_request = device
        self.device = "cpu"
        self.processor: Any = None
        self.model: Any = None

    def load(self) -> None:
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise DependencyUnavailableError("TrOCR requires legopolitics[ocr]") from exc
        self.device = detect_device(self.device_request).device
        self.processor = TrOCRProcessor.from_pretrained(self.model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def recognize(self, image_path: Path, frame_id: str, video_id: str) -> list[OCRRecord]:
        if self.model is None:
            self.load()
        import torch

        try:
            pixels = self.processor(images=to_pil(image_path), return_tensors="pt").pixel_values.to(
                self.device
            )
            with torch.inference_mode():
                ids = self.model.generate(pixels)
            text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            return (
                [
                    OCRRecord(
                        text_region_id=f"ocr_{uuid.uuid4().hex[:16]}",
                        video_id=video_id,
                        frame_id=frame_id,
                        original_text=text,
                        normalized_text=" ".join(text.split()),
                        backend="trocr",
                        confidence=None,
                    )
                ]
                if text
                else []
            )
        except Exception as exc:
            raise ModelInferenceError(f"TrOCR inference failed: {exc}") from exc

    def unload(self) -> None:
        self.processor = None
        self.model = None
