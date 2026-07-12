from __future__ import annotations

from collections import Counter


def percent_agreement(a: list[str], b: list[str]) -> float:
    if len(a) != len(b):
        raise ValueError("Coder vectors must have equal length")
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a) if a else 0.0


def cohens_kappa(a: list[str], b: list[str]) -> float:
    observed = percent_agreement(a, b)
    if not a:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    n = len(a)
    expected = sum((ca[label] / n) * (cb[label] / n) for label in set(ca) | set(cb))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
