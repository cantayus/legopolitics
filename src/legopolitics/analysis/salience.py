from __future__ import annotations

from dataclasses import dataclass

from legopolitics.analysis.spatial import centrality
from legopolitics.schemas import DetectionRecord


@dataclass(frozen=True)
class SalienceComponents:
    screen_area_score: float
    centrality_score: float
    foreground_score: float
    visibility_score: float
    contrast_score: float
    duration_score: float

    @property
    def composite(self) -> float:
        return (
            self.screen_area_score
            * self.centrality_score
            * self.visibility_score
            * self.duration_score
        )


def compute_salience(
    detection: DetectionRecord,
    visibility: float = 1.0,
    contrast: float = 0.5,
    duration: float = 1.0,
) -> SalienceComponents:
    area = min(1.0, (detection.frame_area_fraction or 0.0) * 4)
    center = centrality(detection) or 0.0
    foreground = min(1.0, (detection.frame_area_fraction or 0.0) / 0.15)
    return SalienceComponents(
        area,
        center,
        foreground,
        max(0.0, min(1.0, visibility)),
        max(0.0, min(1.0, contrast)),
        max(0.0, min(1.0, duration)),
    )
