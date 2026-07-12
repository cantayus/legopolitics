from __future__ import annotations

import base64
import json
import os
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from legopolitics.exceptions import ConfigurationError, ModelInferenceError
from legopolitics.schemas import ModelProvenance
from legopolitics.vision.base import ImageInput, VisionRequest, VisionResponseData


class OpenAICompatibleVisionAdapter:
    """Minimal opt-in adapter for OpenAI-compatible chat-completions endpoints."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key_env: str = "OPENAI_API_KEY",
        allow_remote: bool = False,
    ) -> None:
        if not allow_remote:
            raise ConfigurationError("Remote processing requires explicit allow_remote=True")
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    def load(self) -> None:
        if not os.getenv(self.api_key_env):
            raise ConfigurationError(f"Missing credential environment variable: {self.api_key_env}")

    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData:
        self.load()
        path = Path(image) if not hasattr(image, "shape") else None
        if path is None:
            raise ConfigurationError("Remote adapter currently requires an image path")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
            **request.generation,
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {os.environ[self.api_key_env]}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.load(response)
            text = data["choices"][0]["message"]["content"]
            return VisionResponseData(raw_text=text, metadata={"remote": True})
        except Exception as exc:
            raise ModelInferenceError(f"Remote provider request failed: {exc}") from exc

    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]:
        return [self.analyze_image(x, request) for x in images]

    def metadata(self) -> ModelProvenance:
        return ModelProvenance(adapter="openai_compatible", model_id=self.model_id, is_remote=True)

    def unload(self) -> None:
        return
