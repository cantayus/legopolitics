from __future__ import annotations

import uuid

from legopolitics.schemas import Evidence, NarrativeSegment, SceneRecord


def scenes_to_narrative(video_id: str, scenes: list[SceneRecord]) -> list[NarrativeSegment]:
    output: list[NarrativeSegment] = []
    for index, scene in enumerate(scenes):
        label = "introduction" if index == 0 else "other"
        output.append(
            NarrativeSegment(
                narrative_segment_id=f"narrative_{uuid.uuid4().hex[:16]}",
                video_id=video_id,
                label=label,
                start_timestamp=scene.start_timestamp,
                end_timestamp=scene.end_timestamp,
                scene_ids=[scene.scene_id],
                confidence=0.35 if index else 0.50,
                evidence=Evidence(frame_ids=scene.representative_frame_ids),
            )
        )
    return output
