from pathlib import Path

import cv2
import numpy as np
import pytest

from legopolitics.detection import (
    MockDetector,
    fuse_detections,
    iou,
    make_crops,
    register_yolo_weights,
)
from legopolitics.exceptions import ModelLoadError
from legopolitics.schemas import BoundingBox, DetectionRecord


def test_iou_and_fusion():
    a = BoundingBox(x1=0, y1=0, x2=10, y2=10)
    b = BoundingBox(x1=5, y1=5, x2=15, y2=15)
    assert 0 < iou(a, b) < 1
    records = [
        DetectionRecord(
            detection_id="a",
            video_id="v",
            frame_id="f",
            detector_id="d1",
            class_name="figure",
            confidence=0.8,
            box=a,
        ),
        DetectionRecord(
            detection_id="b",
            video_id="v",
            frame_id="f",
            detector_id="d2",
            class_name="figure",
            confidence=0.7,
            box=BoundingBox(x1=1, y1=1, x2=11, y2=11),
        ),
    ]
    fused = fuse_detections(records, 0.5)
    assert len(fused) == 1
    assert set(fused[0].source_detection_ids) == {"a", "b"}


def test_mock_detector_and_crops(tmp_path: Path):
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image_path = tmp_path / "image.jpg"
    cv2.imwrite(str(image_path), image)
    batch = MockDetector().predict([image_path], ["frame_1"], "video_1")[0]
    assert len(batch.detections) == 1
    make_crops(image_path, batch.detections, tmp_path / "crops")
    assert batch.detections[0].crop_path and batch.detections[0].crop_path.exists()


def test_empty_yolo_checkpoint_is_rejected_before_import(tmp_path: Path):
    from legopolitics.detection import YoloDetector
    from legopolitics.exceptions import ModelLoadError

    weights = tmp_path / "best.pt"
    weights.touch()
    detector = YoloDetector(weights)
    try:
        detector.load()
    except ModelLoadError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("An empty YOLO checkpoint must not be accepted")


def test_register_yolo_rejects_empty_checkpoint(tmp_path):
    empty = tmp_path / "best.pt"
    empty.write_bytes(b"")
    with pytest.raises(ModelLoadError, match="empty"):
        register_yolo_weights(empty, tmp_path / "models" / "lego.pt")
