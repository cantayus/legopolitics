from __future__ import annotations

from typing import Any

from legopolitics.exceptions import DependencyUnavailableError


class TranslationAdapter:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise DependencyUnavailableError("Translation requires legopolitics[audio]") from exc
        self.pipeline = pipeline(
            "translation", model=self.model_id, device=0 if self.device == "auto" else self.device
        )

    def translate(self, text: str) -> str:
        if self.pipeline is None:
            self.load()
        result = self.pipeline(text)
        return str(result[0]["translation_text"])
