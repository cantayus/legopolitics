from __future__ import annotations

from typing import Protocol

from legopolitics.schemas import DetectionRecord, FrameRecord, TrackObservation, TrackRecord


class TrackerAdapter(Protocol):
    def track(
        self, frames: list[FrameRecord], detections: list[DetectionRecord]
    ) -> tuple[list[TrackRecord], list[TrackObservation]]: ...
