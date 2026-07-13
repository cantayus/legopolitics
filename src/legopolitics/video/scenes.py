from __future__ import annotations

import cv2

from legopolitics.schemas import SceneRecord
from legopolitics.video.motion import frame_change_score


def detect_scenes(video_path: str, video_id: str, threshold: float = 0.35) -> list[SceneRecord]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    boundaries: list[tuple[int, float]] = [(0, 0.0)]
    previous = None
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        score = frame_change_score(previous, frame)
        if index > 0 and score >= threshold:
            boundaries.append((index, score))
        previous = frame
        index += 1
    capture.release()
    if total <= 0:
        total = index
    boundaries.append((total, 0.0))
    scenes: list[SceneRecord] = []
    for scene_index, ((start, score), (end, _)) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=False)
    ):
        if end <= start:
            continue
        scenes.append(
            SceneRecord(
                scene_id=f"scene_{scene_index:06d}",
                video_id=video_id,
                start_frame_index=start,
                end_frame_index=end - 1,
                start_timestamp=start / fps,
                end_timestamp=(end - 1) / fps,
                change_score=score,
            )
        )
    return scenes
