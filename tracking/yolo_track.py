from __future__ import annotations

from pathlib import Path
from typing import Any

from legopolitics.exceptions import DependencyUnavailableError


class YoloTrackAdapter:
    def __init__(self, weights: Path, tracker: str = "bytetrack.yaml", **kwargs: Any) -> None:
        self.weights = weights
        self.tracker = tracker
        self.kwargs = kwargs

    def track_video(self, video_path: Path) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise DependencyUnavailableError(
                "Native YOLO tracking requires legopolitics[yolo]"
            ) from exc
        model = YOLO(str(self.weights))
        return model.track(
            source=str(video_path), tracker=self.tracker, persist=True, **self.kwargs
        )
