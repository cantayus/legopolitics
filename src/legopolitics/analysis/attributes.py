from __future__ import annotations

from pathlib import Path

import cv2

from legopolitics.analysis.colors import color_statistics


def crop_attributes(path: Path) -> dict[str, float | list[int] | bool]:
    image = cv2.imread(str(path))
    if image is None:
        return {"read_error": True}
    return color_statistics(image)
