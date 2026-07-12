from __future__ import annotations

import uuid
from collections import defaultdict

from legopolitics.detection.postprocess import iou
from legopolitics.schemas import BoundingBox, DetectionRecord


def fuse_detections(
    detections: list[DetectionRecord], iou_threshold: float = 0.55
) -> list[DetectionRecord]:
    grouped: dict[tuple[str, str], list[DetectionRecord]] = defaultdict(list)
    for detection in detections:
        grouped[(detection.frame_id, detection.class_name.casefold())].append(detection)
    fused: list[DetectionRecord] = []
    for (_, _), items in grouped.items():
        pending = sorted(items, key=lambda d: d.confidence, reverse=True)
        while pending:
            seed = pending.pop(0)
            cluster = [seed]
            rest: list[DetectionRecord] = []
            for candidate in pending:
                if iou(seed.box, candidate.box) >= iou_threshold:
                    cluster.append(candidate)
                else:
                    rest.append(candidate)
            pending = rest
            total_weight = sum(max(item.confidence, 1e-6) for item in cluster)
            box = BoundingBox(
                x1=sum(item.box.x1 * item.confidence for item in cluster) / total_weight,
                y1=sum(item.box.y1 * item.confidence for item in cluster) / total_weight,
                x2=sum(item.box.x2 * item.confidence for item in cluster) / total_weight,
                y2=sum(item.box.y2 * item.confidence for item in cluster) / total_weight,
            )
            best = max(cluster, key=lambda d: d.confidence)
            fused.append(
                DetectionRecord(
                    detection_id=f"fused_{uuid.uuid4().hex[:16]}",
                    video_id=best.video_id,
                    frame_id=best.frame_id,
                    detector_id="fusion",
                    class_id=best.class_id,
                    class_name=best.class_name,
                    confidence=max(item.confidence for item in cluster),
                    box=box,
                    normalized_box=best.normalized_box,
                    frame_area_fraction=best.frame_area_fraction,
                    model_id="weighted_box_fusion",
                    source_detection_ids=[item.detection_id for item in cluster],
                )
            )
    return fused
