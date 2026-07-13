from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import ModelProvenance
from legopolitics.utils.device import detect_device
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData
from legopolitics.vision.common import to_pil


class DinoV2Adapter:
    def __init__(self, model_id: str = "facebook/dinov2-large", device: str = "auto") -> None:
        self.model_id = model_id
        self.device_request = device
        self.device = "cpu"
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise DependencyUnavailableError("DINOv2 requires legopolitics[huggingface]") from exc
        self.device = detect_device(self.device_request).device
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def embed_image(self, image: ImageInput) -> np.ndarray:
        if self.model is None:
            self.load()
        import torch

        inputs = self.processor(images=to_pil(image), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        try:
            with torch.inference_mode():
                outputs = self.model(**inputs)
            emb = (
                outputs.pooler_output
                if getattr(outputs, "pooler_output", None) is not None
                else outputs.last_hidden_state[:, 0]
            )
            emb = emb / emb.norm(dim=-1, keepdim=True)
            return emb[0].detach().cpu().numpy()
        except Exception as exc:
            raise ModelInferenceError(f"DINOv2 inference failed: {exc}") from exc

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        emb = self.embed_image(image)
        return VisionResponseData(
            raw_text=f"embedding:{emb.shape[0]}", parsed={"embedding_dimension": int(emb.shape[0])}
        )

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(x, request) for x in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="dinov2", model_id=self.model_id, device=self.device)

    def unload(self) -> None:
        self.model = None
        self.processor = None
