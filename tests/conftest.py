from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    for index in range(40):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.rectangle(image, (10 + index, 25), (50 + index, 100), (0, 255, 255), -1)
        if index >= 20:
            cv2.circle(image, (125, 40), 18, (0, 0, 255), -1)
        writer.write(image)
    writer.release()
    assert path.exists() and path.stat().st_size > 0
    return path
