from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import ModelProvenance
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData
from legopolitics.vision.common import to_pil


class BlipAdapter:
    def __init__(
        self, model_id: str = "Salesforce/blip-image-captioning-large", device: str = "auto"
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise DependencyUnavailableError("BLIP requires legopolitics[huggingface]") from exc
        device = 0 if self.device == "auto" else self.device
        self.pipeline = pipeline("image-to-text", model=self.model_id, device=device)

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        if self.pipeline is None:
            self.load()
        try:
            result = self.pipeline(
                to_pil(image), prompt=request.prompt or None, **request.generation
            )
            text = str(result[0].get("generated_text", result[0]))
            return VisionResponseData(raw_text=text, parsed={"caption": text})
        except Exception as exc:
            raise ModelInferenceError(f"BLIP inference failed: {exc}") from exc

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(x, request) for x in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="blip", model_id=self.model_id, device=self.device)

    def unload(self) -> None:
        self.pipeline = None
