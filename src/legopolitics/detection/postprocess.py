from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from legopolitics.schemas import BoundingBox, DetectionRecord


def iou(a: BoundingBox, b: BoundingBox) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def non_maximum_suppression(
    detections: list[DetectionRecord], threshold: float = 0.5
) -> list[DetectionRecord]:
    remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
    selected: list[DetectionRecord] = []
    while remaining:
        current = remaining.pop(0)
        selected.append(current)
        remaining = [
            item
            for item in remaining
            if item.class_name != current.class_name or iou(item.box, current.box) < threshold
        ]
    return selected


def make_crops(
    image_path: Path,
    detections: list[DetectionRecord],
    crops_dir: Path,
    padding_fraction: float = 0.10,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise OSError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    crops_dir.mkdir(parents=True, exist_ok=True)
    for detection in detections:
        box = detection.box
        pad_x = box.width * padding_fraction
        pad_y = box.height * padding_fraction
        x1 = max(0, int(box.x1 - pad_x))
        y1 = max(0, int(box.y1 - pad_y))
        x2 = min(width, int(box.x2 + pad_x))
        y2 = min(height, int(box.y2 + pad_y))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        target = crops_dir / f"crop_{uuid.uuid4().hex[:12]}_{detection.detection_id}.jpg"
        if cv2.imwrite(str(target), crop):
            detection.crop_path = target
        context_pad_x = box.width * 0.50
        context_pad_y = box.height * 0.50
        cx1 = max(0, int(box.x1 - context_pad_x))
        cy1 = max(0, int(box.y1 - context_pad_y))
        cx2 = min(width, int(box.x2 + context_pad_x))
        cy2 = min(height, int(box.y2 + context_pad_y))
        context = image[cy1:cy2, cx1:cx2]
        context_target = crops_dir / f"context_{detection.detection_id}.jpg"
        if context.size and cv2.imwrite(str(context_target), context):
            detection.context_crop_path = context_target
