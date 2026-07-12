from __future__ import annotations

from pathlib import Path

from legopolitics.schemas.common import SchemaModel


class MaskRecord(SchemaModel):
    mask_id: str
    video_id: str
    frame_id: str
    detection_id: str | None = None
    track_id: str | None = None
    path: Path | None = None
    polygon_json: str | None = None
    rle_json: str | None = None
    area_pixels: int | None = None
    visible_fraction: float | None = None
    touches_border: bool = False
    confidence: float | None = None
