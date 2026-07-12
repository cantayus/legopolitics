from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from legopolitics.exceptions import ConfigurationError, DependencyUnavailableError
from legopolitics.schemas import DiarizationSegment


class PyannoteDiarizationAdapter:
    def __init__(
        self,
        model_id: str = "pyannote/speaker-diarization-community-1",
        token_env: str = "HF_TOKEN",
    ) -> None:
        self.model_id = model_id
        self.token_env = token_env
        self.pipeline: Any = None

    def load(self) -> None:
        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Diarization requires legopolitics[diarization]"
            ) from exc
        token = os.getenv(self.token_env)
        if not token:
            raise ConfigurationError(f"Missing {self.token_env} for gated diarization model")
        self.pipeline = Pipeline.from_pretrained(self.model_id, token=token)

    def diarize(self, audio_path: Path, video_id: str) -> list[DiarizationSegment]:
        if self.pipeline is None:
            self.load()
        annotation = self.pipeline(str(audio_path))
        output = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            output.append(
                DiarizationSegment(
                    diarization_id=f"diar_{uuid.uuid4().hex[:16]}",
                    video_id=video_id,
                    start_timestamp=float(turn.start),
                    end_timestamp=float(turn.end),
                    speaker_label=str(speaker),
                    backend="pyannote",
                )
            )
        return output

    def unload(self) -> None:
        self.pipeline = None
