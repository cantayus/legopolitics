from collections import Counter
from collections.abc import Iterable
from typing import Any


def detection_counts(detections: Iterable[Any]) -> dict[str, int]:
    return dict(Counter(str(d.class_name) for d in detections))
