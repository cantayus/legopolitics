from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def can_reidentify(
    embedding_a: np.ndarray,
    embedding_b: np.ndarray,
    class_a: str,
    class_b: str,
    threshold: float = 0.85,
) -> bool:
    return (
        class_a.casefold() == class_b.casefold()
        and cosine_similarity(embedding_a, embedding_b) >= threshold
    )
