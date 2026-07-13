from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PropagationStatus:
    track_id: str
    start_frame_id: str
    end_frame_id: str
    drift_score: float
    reinitialized: bool
