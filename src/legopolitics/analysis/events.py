from __future__ import annotations

import uuid

from legopolitics.schemas import EventRecord, Evidence, TrackRecord


def track_lifecycle_events(tracks: list[TrackRecord]) -> list[EventRecord]:
    events: list[EventRecord] = []
    for track in tracks:
        events.append(
            EventRecord(
                event_id=f"event_{uuid.uuid4().hex[:16]}",
                video_id=track.video_id,
                label="character_enters",
                start_timestamp=track.first_timestamp,
                end_timestamp=track.first_timestamp,
                participating_track_ids=[track.track_id],
                confidence=1.0,
                evidence=Evidence(frame_ids=[track.first_frame_id], track_ids=[track.track_id]),
            )
        )
        events.append(
            EventRecord(
                event_id=f"event_{uuid.uuid4().hex[:16]}",
                video_id=track.video_id,
                label="character_exits",
                start_timestamp=track.last_timestamp,
                end_timestamp=track.last_timestamp,
                participating_track_ids=[track.track_id],
                confidence=1.0,
                evidence=Evidence(frame_ids=[track.last_frame_id], track_ids=[track.track_id]),
            )
        )
    return events
