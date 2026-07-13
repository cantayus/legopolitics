from __future__ import annotations

import cv2
import numpy as np


def frame_change_score(
    previous: np.ndarray | None,
    current: np.ndarray,
) -> float:
    """Calculate the normalized mean pixel change between two video frames."""
    if previous is None:
        return 0.0

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    previous_resized = cv2.resize(previous_gray, (160, 90))
    current_resized = cv2.resize(current_gray, (160, 90))

    difference = np.asarray(
        cv2.absdiff(previous_resized, current_resized),
        dtype=np.float64,
    )

    return float(difference.mean() / 255.0)
