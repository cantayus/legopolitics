from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from legopolitics.vision.base import ImageInput


def to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image[:, :, ::-1].astype("uint8"))
        return Image.fromarray(image.astype("uint8"))
    return Image.open(Path(image)).convert("RGB")
