from __future__ import annotations

from collections.abc import Sequence

from legopolitics.schemas import ModelProvenance
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData


class MockVisionModel:
    def __init__(self, model_id: str = "mock-vision") -> None:
        self.model_id = model_id

    def load(self) -> None:
        return

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        label = request.candidate_labels[0] if request.candidate_labels else None
        parsed = {
            "label": label,
            "observation": "Synthetic mock visual response",
            "task": request.task,
        }
        return VisionResponseData(
            raw_text="Synthetic mock visual response", parsed=parsed, confidence=0.5
        )

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(image, request) for image in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="mock_vision", model_id=self.model_id)

    def unload(self) -> None:
        return
