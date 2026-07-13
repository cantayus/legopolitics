from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from legopolitics.schemas import DetectionRecord, MaskRecord, ModelProvenance


class SegmenterAdapter(Protocol):
    def load(self) -> None: ...
    def segment(
        self, image_path: Path, detections: Sequence[DetectionRecord], masks_dir: Path
    ) -> list[MaskRecord]: ...
    def metadata(self) -> ModelProvenance: ...
    def unload(self) -> None: ...
