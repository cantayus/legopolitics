from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field

from legopolitics.detection.postprocess import iou
from legopolitics.schemas import DetectionRecord, FrameRecord, TrackObservation, TrackRecord


@dataclass
class _ActiveTrack:
    track_id: str
    class_name: str
    last_detection: DetectionRecord
    last_frame_position: int
    detections: list[tuple[DetectionRecord, FrameRecord]] = field(default_factory=list)


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.30, maximum_gap_frames: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.maximum_gap_frames = maximum_gap_frames

    def track(
        self, frames: list[FrameRecord], detections: list[DetectionRecord]
    ) -> tuple[list[TrackRecord], list[TrackObservation]]:
        position = {frame.frame_id: i for i, frame in enumerate(frames)}
        by_frame: dict[str, list[DetectionRecord]] = {frame.frame_id: [] for frame in frames}
        for detection in detections:
            by_frame.setdefault(detection.frame_id, []).append(detection)
        active: list[_ActiveTrack] = []
        complete: list[_ActiveTrack] = []
        observations: list[TrackObservation] = []
        for frame in frames:
            pos = position[frame.frame_id]
            current = by_frame.get(frame.frame_id, [])
            available = set(range(len(active)))
            for detection in sorted(current, key=lambda d: d.confidence, reverse=True):
                best_idx = None
                best_score = self.iou_threshold
                for idx in available:
                    candidate = active[idx]
                    if candidate.class_name.casefold() != detection.class_name.casefold():
                        continue
                    if pos - candidate.last_frame_position > self.maximum_gap_frames:
                        continue
                    score = iou(candidate.last_detection.box, detection.box)
                    if score >= best_score:
                        best_idx, best_score = idx, score
                if best_idx is None:
                    track = _ActiveTrack(
                        track_id=f"track_{uuid.uuid4().hex[:16]}",
                        class_name=detection.class_name,
                        last_detection=detection,
                        last_frame_position=pos,
                        detections=[(detection, frame)],
                    )
                    active.append(track)
                else:
                    track = active[best_idx]
                    track.last_detection = detection
                    track.last_frame_position = pos
                    track.detections.append((detection, frame))
                    available.remove(best_idx)
                observations.append(
                    TrackObservation(
                        track_id=track.track_id,
                        frame_id=frame.frame_id,
                        detection_id=detection.detection_id,
                        timestamp_seconds=frame.timestamp_seconds,
                        confidence=detection.confidence,
                        screen_area_fraction=detection.frame_area_fraction,
                        centrality=_centrality(detection),
                    )
                )
            still_active: list[_ActiveTrack] = []
            for track in active:
                if pos - track.last_frame_position > self.maximum_gap_frames:
                    complete.append(track)
                else:
                    still_active.append(track)
            active = still_active
        complete.extend(active)
        tracks = [_summarize_track(track, frames) for track in complete]
        return tracks, observations


def _centrality(detection: DetectionRecord) -> float | None:
    box = detection.normalized_box
    if box is None:
        return None
    cx, cy = box.center
    distance = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    return max(0.0, 1.0 - distance / (2**0.5 / 2))


def _summarize_track(track: _ActiveTrack, frames: list[FrameRecord]) -> TrackRecord:
    pairs = track.detections
    detections = [item[0] for item in pairs]
    frame_records = [item[1] for item in pairs]
    confidences = [d.confidence for d in detections]
    areas = [d.frame_area_fraction for d in detections if d.frame_area_fraction is not None]
    centralities = [c for d in detections if (c := _centrality(d)) is not None]
    first, last = frame_records[0], frame_records[-1]
    duration = max(0.0, last.timestamp_seconds - first.timestamp_seconds)
    step = (
        statistics.median(
            [
                b.timestamp_seconds - a.timestamp_seconds
                for a, b in zip(frame_records[:-1], frame_records[1:], strict=False)
            ]
        )
        if len(frame_records) > 1
        else 0.0
    )
    visible_duration = duration + max(0.0, step)
    representative = max(detections, key=lambda d: (d.frame_area_fraction or 0) * d.confidence)
    return TrackRecord(
        track_id=track.track_id,
        video_id=detections[0].video_id,
        object_type=track.class_name,
        first_frame_id=first.frame_id,
        last_frame_id=last.frame_id,
        first_timestamp=first.timestamp_seconds,
        last_timestamp=last.timestamp_seconds,
        observed_frame_count=len(detections),
        estimated_duration=duration,
        visible_duration=visible_duration,
        screen_time=visible_duration,
        mean_confidence=statistics.mean(confidences),
        minimum_confidence=min(confidences),
        maximum_confidence=max(confidences),
        mean_screen_area=statistics.mean(areas) if areas else None,
        maximum_screen_area=max(areas) if areas else None,
        mean_centrality=statistics.mean(centralities) if centralities else None,
        representative_crop_path=representative.crop_path,
        track_quality_score=min(1.0, len(detections) / 5.0) * statistics.mean(confidences),
    )
