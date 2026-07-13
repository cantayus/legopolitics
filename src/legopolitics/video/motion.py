from __future__ import annotations

import cv2
import numpy as np


def frame_change_score(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    prev = cv2.resize(cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY), (160, 90))
    curr = cv2.resize(cv2.cvtColor(current, cv2.COLOR_BGR2GRAY), (160, 90))
    return float(np.mean(cv2.absdiff(prev, curr)) / 255.0)
