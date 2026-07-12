from __future__ import annotations

import uuid
from collections import defaultdict

from legopolitics.schemas import Evidence, RelationRecord, TrackObservation


def infer_proximity_relations(
    observations: list[TrackObservation], video_id: str
) -> list[RelationRecord]:
    by_frame: dict[str, list[TrackObservation]] = defaultdict(list)
    for item in observations:
        by_frame[item.frame_id].append(item)
    relations: list[RelationRecord] = []
    for frame_id, items in by_frame.items():
        for i, first in enumerate(items):
            for second in items[i + 1 :]:
                relations.append(
                    RelationRecord(
                        relation_id=f"rel_{uuid.uuid4().hex[:16]}",
                        video_id=video_id,
                        start_timestamp=min(first.timestamp_seconds, second.timestamp_seconds),
                        end_timestamp=max(first.timestamp_seconds, second.timestamp_seconds),
                        subject_track_id=first.track_id,
                        predicate="co_occurs_with",
                        object_track_id=second.track_id,
                        confidence=0.50,
                        evidence=Evidence(
                            frame_ids=[frame_id], track_ids=[first.track_id, second.track_id]
                        ),
                    )
                )
    return relations
