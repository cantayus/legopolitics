from __future__ import annotations

import statistics
import uuid
from collections import Counter, defaultdict

from legopolitics.analysis.uncertainty import entropy
from legopolitics.schemas import ModelAgreementRecord, VisualResponse


def aggregate_model_responses(
    video_id: str, responses: list[VisualResponse]
) -> list[ModelAgreementRecord]:
    by_unit: dict[tuple[str, str], list[VisualResponse]] = defaultdict(list)
    for response in responses:
        by_unit[(response.target_type, response.target_id)].append(response)
    output: list[ModelAgreementRecord] = []
    for (unit_type, unit_id), items in by_unit.items():
        labels = [
            str(item.parsed.get("label"))
            for item in items
            if item.parsed and item.parsed.get("label") is not None
        ]
        confidences = [item.confidence for item in items if item.confidence is not None]
        majority = Counter(labels).most_common(1)[0][0] if labels else None
        max_item = max(items, key=lambda r: r.confidence or -1, default=None)
        max_label = (
            str(max_item.parsed.get("label"))
            if max_item and max_item.parsed and max_item.parsed.get("label") is not None
            else None
        )
        support = Counter(labels)[majority] if majority else 0
        output.append(
            ModelAgreementRecord(
                agreement_id=f"agree_{uuid.uuid4().hex[:16]}",
                video_id=video_id,
                unit_type=unit_type,
                unit_id=unit_id,
                model_count=len(items),
                support_count=support,
                majority_label=majority,
                maximum_confidence_label=max_label,
                weighted_label=majority,
                mean_confidence=statistics.mean(confidences) if confidences else None,
                median_confidence=statistics.median(confidences) if confidences else None,
                minimum_confidence=min(confidences) if confidences else None,
                maximum_confidence=max(confidences) if confidences else None,
                entropy=entropy(labels),
                disagreement_flag=len(set(labels)) > 1,
                human_review_priority=min(
                    1.0,
                    entropy(labels) / 2.0
                    + (1 - (statistics.mean(confidences) if confidences else 0.5)) / 2,
                ),
            )
        )
    return output
