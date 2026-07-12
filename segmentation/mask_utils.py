from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def save_mask(mask: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    binary = (mask > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), binary):
        raise OSError(f"Could not write mask: {path}")
    return path


def mask_to_polygons(mask: np.ndarray) -> str:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = [contour.reshape(-1, 2).tolist() for contour in contours if len(contour) >= 3]
    return json.dumps(polygons)


def touches_border(mask: np.ndarray) -> bool:
    binary = mask > 0
    return bool(binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any())
