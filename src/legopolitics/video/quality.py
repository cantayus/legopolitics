from __future__ import annotations

import cv2
import numpy as np

from legopolitics.schemas import FrameQuality


def analyze_frame_quality(image: np.ndarray) -> FrameQuality:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = float(gray.mean() / 255.0)
    contrast = float(gray.std() / 127.5)
    saturation = float(hsv[:, :, 1].mean() / 255.0)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(1.0 / (1.0 + lap_var / 100.0))
    black = float(np.mean(gray < 10))
    under = float(np.mean(gray < 35))
    over = float(np.mean(gray > 220))
    block_edges = 0.0
    if gray.shape[0] > 8 and gray.shape[1] > 8:
        vertical = np.abs(np.diff(gray.astype(np.float32), axis=1))
        horizontal = np.abs(np.diff(gray.astype(np.float32), axis=0))
        block_v = vertical[:, 7::8].mean() if vertical[:, 7::8].size else 0.0
        block_h = horizontal[7::8, :].mean() if horizontal[7::8, :].size else 0.0
        block_edges = float(min(1.0, (block_v + block_h) / 100.0))
    readability = float(max(0.0, min(1.0, (1.0 - blur_score) * (1.0 - black) * (1.0 - over))))
    suitability = float(max(0.0, min(1.0, readability * (0.5 + 0.5 * min(1.0, contrast)))))
    return FrameQuality(
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        black_frame_score=black,
        underexposure_score=under,
        overexposure_score=over,
        compression_artifact_score=block_edges,
        readability_score=readability,
        crop_suitability_score=suitability,
    )
