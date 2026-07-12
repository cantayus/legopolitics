from __future__ import annotations

import statistics
from collections import Counter

from legopolitics.analysis.spatial import centrality
from legopolitics.schemas import DetectionRecord


def frame_composition(detections: list[DetectionRecord]) -> dict[str, object]:
    counts = Counter(d.class_name for d in detections)
    areas = [d.frame_area_fraction or 0.0 for d in detections]
    largest = max(detections, key=lambda d: d.frame_area_fraction or 0.0, default=None)
    center = max(detections, key=lambda d: centrality(d) or 0.0, default=None)
    return {
        "object_count": len(detections),
        "class_counts": dict(counts),
        "mean_object_size": statistics.mean(areas) if areas else 0.0,
        "largest_object_id": largest.detection_id if largest else None,
        "center_most_object_id": center.detection_id if center else None,
        "visual_crowding": min(1.0, sum(areas)),
    }
