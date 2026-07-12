from __future__ import annotations

import statistics
from collections import Counter


def vote(
    predictions: list[tuple[str, float, float]],
    method: str = "majority",
    consensus_threshold: float = 0.5,
) -> dict[str, object]:
    if not predictions:
        return {"label": None, "abstained": True, "support": 0}
    labels = [p[0] for p in predictions]
    counts = Counter(labels)
    if method == "maximum_confidence":
        label = max(predictions, key=lambda x: x[1])[0]
    elif method in {"weighted", "reliability_weighted"}:
        scores: dict[str, float] = {}
        for label, confidence, reliability in predictions:
            scores[label] = scores.get(label, 0) + confidence * (
                reliability if method == "reliability_weighted" else 1
            )
        label = max(scores, key=lambda candidate: scores[candidate])
    else:
        label = counts.most_common(1)[0][0]
    support = counts[label] / len(predictions)
    confidences = [p[1] for p in predictions if p[0] == label]
    return {
        "label": label if support >= consensus_threshold else None,
        "candidate_label": label,
        "abstained": support < consensus_threshold,
        "support": support,
        "mean_confidence": statistics.mean(confidences) if confidences else None,
    }
