from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import ModelProvenance
from legopolitics.utils.device import detect_device, resolve_torch_dtype
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData
from legopolitics.vision.common import to_pil


class Blip2Adapter:
    def __init__(
        self, model_id: str = "Salesforce/blip2-opt-2.7b", device: str = "auto", dtype: str = "auto"
    ) -> None:
        self.model_id = model_id
        self.device_request = device
        self.dtype_name = dtype
        self.device = "cpu"
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        try:
            from transformers import Blip2ForConditionalGeneration, Blip2Processor
        except ImportError as exc:
            raise DependencyUnavailableError("BLIP-2 requires legopolitics[huggingface]") from exc
        info = detect_device(self.device_request)
        self.device = info.device
        dtype = resolve_torch_dtype(self.dtype_name, self.device)
        self.processor = Blip2Processor.from_pretrained(self.model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        if self.model is None:
            self.load()
        import torch

        try:
            inputs = self.processor(
                images=to_pil(image), text=request.prompt or None, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=256, **request.generation)
            text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            return VisionResponseData(raw_text=text, parsed={"caption": text})
        except Exception as exc:
            raise ModelInferenceError(f"BLIP-2 inference failed: {exc}") from exc

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(x, request) for x in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(
            adapter="blip2", model_id=self.model_id, device=self.device, dtype=self.dtype_name
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
