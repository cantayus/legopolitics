from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError
from legopolitics.schemas import AudioEvent


class AudioEventAdapter:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Audio event classification requires legopolitics[audio]"
            ) from exc
        self.pipeline = pipeline(
            "audio-classification",
            model=self.model_id,
            device=0 if self.device == "auto" else self.device,
        )

    def classify(self, audio_path: Path, video_id: str) -> list[AudioEvent]:
        if self.pipeline is None:
            self.load()
        results = self.pipeline(str(audio_path), top_k=None)
        return [
            AudioEvent(
                audio_event_id=f"audioevt_{uuid.uuid4().hex[:16]}",
                video_id=video_id,
                start_timestamp=0,
                end_timestamp=0,
                label=str(r["label"]),
                confidence=float(r["score"]),
                backend=self.model_id,
            )
            for r in results
        ]

    def unload(self) -> None:
        self.pipeline = None
