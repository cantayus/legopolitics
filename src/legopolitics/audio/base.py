from __future__ import annotations

from pathlib import Path
from typing import Protocol

from legopolitics.schemas import TranscriptSegment


class ASRAdapter(Protocol):
    def load(self) -> None: ...
    def transcribe(self, audio_path: Path, video_id: str) -> list[TranscriptSegment]: ...
    def unload(self) -> None: ...
