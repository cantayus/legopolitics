from __future__ import annotations

import random
import uuid

from legopolitics.schemas import AnalysisResult, ValidationRecord


def sample_for_validation(
    result: AnalysisResult,
    random_size: int = 100,
    low_confidence_size: int = 100,
    disagreement_size: int = 100,
    seed: int = 2026,
) -> list[ValidationRecord]:
    rng = random.Random(seed)
    records = []
    detections = list(result.detections)
    rng.shuffle(detections)
    for detection in detections[:random_size]:
        records.append(
            ValidationRecord(
                validation_id=f"val_{uuid.uuid4().hex[:16]}",
                video_id=result.video.video_id,
                unit_type="detection",
                unit_id=detection.detection_id,
                sampling_reason="random",
                suggested_label=detection.class_name,
                suggested_confidence=detection.confidence,
            )
        )
    for detection in sorted(result.detections, key=lambda d: d.confidence)[:low_confidence_size]:
        records.append(
            ValidationRecord(
                validation_id=f"val_{uuid.uuid4().hex[:16]}",
                video_id=result.video.video_id,
                unit_type="detection",
                unit_id=detection.detection_id,
                sampling_reason="low_confidence",
                suggested_label=detection.class_name,
                suggested_confidence=detection.confidence,
            )
        )
    for agreement in sorted(
        [a for a in result.model_agreement if a.disagreement_flag],
        key=lambda a: a.human_review_priority or 0,
        reverse=True,
    )[:disagreement_size]:
        records.append(
            ValidationRecord(
                validation_id=f"val_{uuid.uuid4().hex[:16]}",
                video_id=result.video.video_id,
                unit_type=agreement.unit_type,
                unit_id=agreement.unit_id,
                sampling_reason="model_disagreement",
                suggested_label=agreement.majority_label,
                suggested_confidence=agreement.mean_confidence,
            )
        )
    seen = set()
    unique = []
    for record in records:
        key = (record.unit_type, record.unit_id, record.sampling_reason)
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique
