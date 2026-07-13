from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, TypeAlias

import numpy as np

from legopolitics.schemas import DetectionBatch, ModelProvenance

ImageInput: TypeAlias = str | Path | np.ndarray


class DetectorAdapter(Protocol):
    def load(self) -> None: ...
    def predict(
        self, images: Sequence[ImageInput], frame_ids: Sequence[str], video_id: str
    ) -> list[DetectionBatch]: ...
    def metadata(self) -> ModelProvenance: ...
    def unload(self) -> None: ...
