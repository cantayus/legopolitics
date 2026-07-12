from __future__ import annotations

from legopolitics.schemas import DetectionRecord


def centrality(detection: DetectionRecord) -> float | None:
    box = detection.normalized_box
    if box is None:
        return None
    cx, cy = box.center
    max_distance = (0.5**2 + 0.5**2) ** 0.5
    return max(0.0, 1.0 - (((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5 / max_distance))


def horizontal_position(detection: DetectionRecord) -> str | None:
    box = detection.normalized_box
    if box is None:
        return None
    cx, _ = box.center
    return "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"


def depth_proxy(detection: DetectionRecord) -> str | None:
    area = detection.frame_area_fraction
    if area is None:
        return None
    return "foreground" if area >= 0.15 else "background" if area <= 0.03 else "middle_ground"
