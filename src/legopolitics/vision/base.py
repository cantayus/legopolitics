from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, TypeAlias

import numpy as np
from pydantic import BaseModel, Field

from legopolitics.schemas import ModelProvenance

ImageInput: TypeAlias = str | Path | np.ndarray


class VisionRequest(BaseModel):
    prompt: str
    task: str = "caption"
    candidate_labels: list[str] = Field(default_factory=list)
    generation: dict[str, Any] = Field(default_factory=dict)
    response_schema: dict[str, Any] | None = None


class VisionResponseData(BaseModel):
    raw_text: str
    parsed: dict[str, Any] | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionModelAdapter(Protocol):
    def load(self) -> None: ...
    def analyze_image(self, image: ImageInput, request: VisionRequest) -> VisionResponseData: ...
    def analyze_images(
        self, images: Sequence[ImageInput], request: VisionRequest
    ) -> list[VisionResponseData]: ...
    def metadata(self) -> ModelProvenance: ...
    def unload(self) -> None: ...
