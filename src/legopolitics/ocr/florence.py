from __future__ import annotations

import uuid
from pathlib import Path

from legopolitics.schemas import OCRRecord
from legopolitics.vision.base import VisionRequest
from legopolitics.vision.florence2 import Florence2Adapter


class FlorenceOCRAdapter:
    def __init__(
        self, model_id: str = "microsoft/Florence-2-large-ft", device: str = "auto"
    ) -> None:
        self.adapter = Florence2Adapter(model_id=model_id, device=device)

    def load(self) -> None:
        self.adapter.load()

    def recognize(self, image_path: Path, frame_id: str, video_id: str) -> list[OCRRecord]:
        result = self.adapter.analyze_image(
            image_path, VisionRequest(prompt="", task="ocr_with_region")
        )
        parsed = result.parsed or {}
        texts = []
        if isinstance(parsed, dict):
            value = next(iter(parsed.values()), parsed)
            if isinstance(value, dict) and isinstance(value.get("labels"), list):
                texts = value["labels"]
            elif isinstance(value, str):
                texts = [value]
        if not texts and result.raw_text.strip():
            texts = [result.raw_text.strip()]
        return [
            OCRRecord(
                text_region_id=f"ocr_{uuid.uuid4().hex[:16]}",
                video_id=video_id,
                frame_id=frame_id,
                original_text=t,
                normalized_text=" ".join(t.split()),
                backend="florence2",
            )
            for t in texts
            if t
        ]

    def unload(self) -> None:
        self.adapter.unload()
