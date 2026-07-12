from __future__ import annotations

import cv2
import numpy as np


def color_statistics(image: np.ndarray) -> dict[str, float | list[int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = rgb.reshape(-1, 3)
    dominant = np.median(pixels, axis=0).astype(int).tolist()
    red = ((rgb[:, :, 0] > 140) & (rgb[:, :, 0] > rgb[:, :, 1] * 1.25)).mean()
    green = ((rgb[:, :, 1] > 100) & (rgb[:, :, 1] > rgb[:, :, 0] * 1.15)).mean()
    black = (rgb.mean(axis=2) < 35).mean()
    return {
        "dominant_rgb": dominant,
        "brightness": float(hsv[:, :, 2].mean() / 255),
        "saturation": float(hsv[:, :, 1].mean() / 255),
        "red_proportion": float(red),
        "green_proportion": float(green),
        "black_proportion": float(black),
    }
