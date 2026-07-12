from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError, ModelInferenceError
from legopolitics.schemas import TranscriptSegment


class WhisperAdapter:
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3",
        device: str = "auto",
        language: str | None = None,
        task: str = "transcribe",
        word_timestamps: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.language = language
        self.task = task
        self.word_timestamps = word_timestamps
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise DependencyUnavailableError("Whisper requires legopolitics[audio]") from exc
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=0 if self.device == "auto" else self.device,
        )

    def transcribe(self, audio_path: Path, video_id: str) -> list[TranscriptSegment]:
        if self.pipeline is None:
            self.load()
        kwargs: dict[str, Any] = {"return_timestamps": "word" if self.word_timestamps else True}
        generate_kwargs = {"task": self.task}
        if self.language:
            generate_kwargs["language"] = self.language
        kwargs["generate_kwargs"] = generate_kwargs
        try:
            result = self.pipeline(str(audio_path), **kwargs)
        except Exception as exc:
            raise ModelInferenceError(f"Whisper transcription failed: {exc}") from exc
        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        if chunks:
            segments = []
            for chunk in chunks:
                timestamp = chunk.get("timestamp", (0.0, 0.0))
                start = float(timestamp[0] or 0)
                end = float(timestamp[1] or start)
                segments.append(
                    TranscriptSegment(
                        segment_id=f"asr_{uuid.uuid4().hex[:16]}",
                        video_id=video_id,
                        start_timestamp=start,
                        end_timestamp=end,
                        original_text=str(chunk.get("text", "")).strip(),
                        language=result.get("language"),
                        backend="whisper",
                        model_revision=self.model_id,
                    )
                )
            return segments
        text = str(result.get("text", "")).strip() if isinstance(result, dict) else str(result)
        return (
            [
                TranscriptSegment(
                    segment_id=f"asr_{uuid.uuid4().hex[:16]}",
                    video_id=video_id,
                    start_timestamp=0,
                    end_timestamp=0,
                    original_text=text,
                    language=result.get("language") if isinstance(result, dict) else None,
                    backend="whisper",
                    model_revision=self.model_id,
                )
            ]
            if text
            else []
        )

    def unload(self) -> None:
        self.pipeline = None
