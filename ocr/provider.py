from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from legopolitics.exceptions import ConfigurationError, ModelInferenceError
from legopolitics.schemas import BoundingBox, ModelProvenance, OCRRecord

ProviderResult = dict[str, Any]
ProviderCallable = Callable[[Path], list[ProviderResult]]


class ProviderOCRAdapter:
    """Adapt an explicitly supplied OCR callable to the package OCR protocol.

    The callable receives a local image path and returns dictionaries containing
    at least ``text``. Optional keys include ``confidence``, ``language``,
    ``translated_text``, ``text_type``, and ``box`` as ``[x1, y1, x2, y2]``.
    Remote providers require explicit opt-in so images are never uploaded
    silently.
    """

    def __init__(
        self,
        provider: ProviderCallable,
        model_id: str,
        *,
        is_remote: bool = False,
        allow_remote: bool = False,
    ) -> None:
        if is_remote and not allow_remote:
            raise ConfigurationError("Remote OCR requires explicit allow_remote=True")
        self.provider = provider
        self.model_id = model_id
        self.is_remote = is_remote

    def recognize(self, image_path: Path, frame_id: str, video_id: str) -> list[OCRRecord]:
        try:
            candidates = self.provider(Path(image_path))
        except Exception as exc:
            raise ModelInferenceError(f"Provider OCR failed: {exc}") from exc

        records: list[OCRRecord] = []
        for candidate in candidates:
            text = str(candidate.get("text", "")).strip()
            if not text:
                continue
            box_value = candidate.get("box")
            box = (
                BoundingBox(
                    x1=float(box_value[0]),
                    y1=float(box_value[1]),
                    x2=float(box_value[2]),
                    y2=float(box_value[3]),
                )
                if box_value
                else None
            )
            records.append(
                OCRRecord(
                    text_region_id=f"ocr_{uuid.uuid4().hex[:16]}",
                    video_id=video_id,
                    frame_id=frame_id,
                    bounding_box=box,
                    original_text=text,
                    normalized_text=" ".join(text.split()),
                    detected_language=candidate.get("language"),
                    translated_text=candidate.get("translated_text"),
                    confidence=candidate.get("confidence"),
                    backend=self.model_id,
                    text_type=candidate.get("text_type"),
                )
            )
        return records

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(
            adapter="provider_ocr",
            model_id=self.model_id,
            is_remote=self.is_remote,
        )

    def unload(self) -> None:
        """Release adapter resources; provider callables own their own lifecycle."""
