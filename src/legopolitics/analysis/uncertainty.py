from __future__ import annotations

import math
import statistics
from collections import Counter


def entropy(labels: list[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def confidence_summary(confidences: list[float]) -> dict[str, float | None]:
    if not confidences:
        return {"mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "mean": statistics.mean(confidences),
        "median": statistics.median(confidences),
        "minimum": min(confidences),
        "maximum": max(confidences),
    }
