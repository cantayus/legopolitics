from __future__ import annotations

from pathlib import Path
from typing import Protocol

from legopolitics.schemas import OCRRecord


class OCRAdapter(Protocol):
    def load(self) -> None: ...
    def recognize(self, image_path: Path, frame_id: str, video_id: str) -> list[OCRRecord]: ...
    def unload(self) -> None: ...
