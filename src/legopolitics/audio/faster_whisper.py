from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError
from legopolitics.schemas import TranscriptSegment


class FasterWhisperAdapter:
    def __init__(
        self,
        model_id: str = "large-v3",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
        task: str = "transcribe",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.task = task
        self.model: Any = None

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise DependencyUnavailableError("Faster-Whisper requires legopolitics[audio]") from exc
        device = "cuda" if self.device == "auto" else self.device
        self.model = WhisperModel(self.model_id, device=device, compute_type=self.compute_type)

    def transcribe(self, audio_path: Path, video_id: str) -> list[TranscriptSegment]:
        if self.model is None:
            self.load()
        segments, info = self.model.transcribe(
            str(audio_path), language=self.language, task=self.task, word_timestamps=True
        )
        return [
            TranscriptSegment(
                segment_id=f"asr_{uuid.uuid4().hex[:16]}",
                video_id=video_id,
                start_timestamp=float(s.start),
                end_timestamp=float(s.end),
                original_text=s.text.strip(),
                language=info.language,
                confidence=getattr(s, "avg_logprob", None),
                words=[
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (s.words or [])
                ],
                backend="faster_whisper",
                model_revision=self.model_id,
            )
            for s in segments
        ]

    def unload(self) -> None:
        self.model = None
