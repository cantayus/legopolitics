from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import ModelProvenance
from legopolitics.utils.device import detect_device
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData
from legopolitics.vision.common import to_pil


class ClipAdapter:
    def __init__(
        self, model_id: str = "openai/clip-vit-large-patch14", device: str = "auto"
    ) -> None:
        self.model_id = model_id
        self.device_request = device
        self.device = "cpu"
        self.model: Any = None
        self.processor: Any = None

    def load(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise DependencyUnavailableError("CLIP requires legopolitics[huggingface]") from exc
        self.device = detect_device(self.device_request).device
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        if self.model is None:
            self.load()
        import torch

        labels = request.candidate_labels
        if not labels:
            raise ValueError("CLIP requires candidate_labels")
        try:
            inputs = self.processor(
                text=labels, images=to_pil(image), return_tensors="pt", padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu().tolist()
            ranked = sorted(zip(labels, probs, strict=True), key=lambda x: x[1], reverse=True)
            return VisionResponseData(
                raw_text=str(ranked),
                parsed={"label": ranked[0][0], "scores": dict(ranked)},
                confidence=float(ranked[0][1]),
            )
        except Exception as exc:
            raise ModelInferenceError(f"CLIP inference failed: {exc}") from exc

    def embed_image(self, image: ImageInput) -> np.ndarray:
        if self.model is None:
            self.load()
        import torch

        inputs = self.processor(images=to_pil(image), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].detach().cpu().numpy()

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(x, request) for x in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="clip", model_id=self.model_id, device=self.device)

    def unload(self) -> None:
        self.model = None
        self.processor = None
