from __future__ import annotations

from pathlib import Path
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError
from legopolitics.schemas import ModelProvenance
from legopolitics.vision.base import VisionRequest, VisionResponseData


class VideoVLMAdapter:
    def __init__(self, model_id: str, backend: str = "huggingface", device: str = "auto") -> None:
        self.model_id = model_id
        self.backend = backend
        self.device = device
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Video VLM support requires legopolitics[huggingface]"
            ) from exc
        self.pipeline = pipeline(
            "video-text-to-text",
            model=self.model_id,
            device=0 if self.device == "auto" else self.device,
        )

    def analyze_video(self, video_path: Path, request: VisionRequest) -> VisionResponseData:
        if self.pipeline is None:
            self.load()
        result = self.pipeline(video=str(video_path), text=request.prompt, **request.generation)
        text = str(
            result[0].get("generated_text", result[0]) if isinstance(result, list) else result
        )
        return VisionResponseData(raw_text=text)

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="video_vlm", model_id=self.model_id, device=self.device)

    def unload(self) -> None:
        self.pipeline = None
