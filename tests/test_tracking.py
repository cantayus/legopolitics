from legopolitics.schemas import BoundingBox, DetectionRecord, FrameRecord
from legopolitics.tracking import IoUTracker


def test_iou_tracker_continuity():
    frames = [
        FrameRecord(
            video_id="v",
            frame_id=f"f{i}",
            source_frame_index=i,
            timestamp_seconds=i,
            timestamp_ms=i * 1000,
            source_fps=1.0,
        )
        for i in range(3)
    ]
    detections = [
        DetectionRecord(
            detection_id=f"d{i}",
            video_id="v",
            frame_id=f"f{i}",
            detector_id="x",
            class_name="figure",
            confidence=0.9,
            box=BoundingBox(x1=i, y1=0, x2=10 + i, y2=10),
            frame_area_fraction=0.1,
        )
        for i in range(3)
    ]
    tracks, observations = IoUTracker(0.2, 2).track(frames, detections)
    assert len(tracks) == 1
    assert tracks[0].observed_frame_count == 3
    assert len({item.track_id for item in observations}) == 1
