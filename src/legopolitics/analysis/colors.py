from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt


def color_statistics(
    image: np.ndarray,
) -> dict[str, float | list[int]]:
    """Calculate basic color statistics for an image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixels: npt.NDArray[np.float64] = np.asarray(
        rgb.reshape(-1, 3),
        dtype=np.float64,
    )

    dominant_array = np.asarray(
        np.median(pixels, axis=0),
        dtype=np.int64,
    )
    dominant = [int(value) for value in dominant_array.tolist()]

    red_mask = (rgb[:, :, 0] > 140) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.25)
    green_mask = (rgb[:, :, 1] > 100) & (rgb[:, :, 1] > rgb[:, :, 0] * 1.15)
    black_mask = rgb.mean(axis=2) < 35

    return {
        "dominant_rgb": dominant,
        "brightness": float(hsv[:, :, 2].mean() / 255.0),
        "saturation": float(hsv[:, :, 1].mean() / 255.0),
        "red_proportion": float(red_mask.mean()),
        "green_proportion": float(green_mask.mean()),
        "black_proportion": float(black_mask.mean()),
    }
