from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from legopolitics.exceptions import VideoDecodeError


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, float, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoDecodeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield index, index / fps, frame
            index += 1
    finally:
        capture.release()


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise VideoDecodeError(f"Could not read image: {path}")
    return image
