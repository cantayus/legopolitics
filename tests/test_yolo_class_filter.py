from __future__ import annotations

from types import SimpleNamespace

import pytest

from legopolitics.detection.yolo import YoloDetector
from legopolitics.exceptions import ModelLoadError


def test_yolo_class_filter_accepts_names_and_ids(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    detector = YoloDetector(weights, class_filter=["lego_figure", 0, "LEGO_FIGURE"])
    detector.model = SimpleNamespace(names={0: "human", 1: "lego_figure"})
    assert detector._resolve_class_filter() == [0, 1]


def test_yolo_class_filter_rejects_unknown_name(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    detector = YoloDetector(weights, class_filter=["vehicle"])
    detector.model = SimpleNamespace(names={0: "human", 1: "lego_figure"})
    with pytest.raises(ModelLoadError, match="not present"):
        detector._resolve_class_filter()
