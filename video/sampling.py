from __future__ import annotations

import logging
from pathlib import Path

import cv2

from legopolitics.config import AnalysisConfig
from legopolitics.exceptions import VideoDecodeError
from legopolitics.schemas import FrameRecord, SceneRecord, VideoRecord
from legopolitics.video.hashing import difference_hash, hamming_distance
from legopolitics.video.motion import frame_change_score
from legopolitics.video.quality import analyze_frame_quality
from legopolitics.video.scenes import detect_scenes

LOGGER = logging.getLogger("legopolitics")


def _scene_for_frame(index: int, scenes: list[SceneRecord]) -> str | None:
    for scene in scenes:
        if scene.start_frame_index <= index <= scene.end_frame_index:
            return scene.scene_id
    return None


def _candidate_indices(
    video: VideoRecord, scenes: list[SceneRecord], config: AnalysisConfig
) -> dict[int, set[str]]:
    fps = video.average_fps or video.nominal_fps or 25.0
    total = video.frame_count or int((video.duration_seconds or 0) * fps)
    reasons: dict[int, set[str]] = {}

    def add(index: int, reason: str) -> None:
        if total > 0:
            index = max(0, min(total - 1, index))
        reasons.setdefault(index, set()).add(reason)

    mode = config.sampling.mode
    if mode in {"fixed", "hybrid", "adaptive"}:
        step = max(1, round(fps / config.sampling.fps))
        for index in range(0, max(total, 1), step):
            add(index, "fixed")
    if mode in {"scene", "hybrid", "adaptive"}:
        for scene in scenes:
            add(scene.start_frame_index, "scene_start")
            add(scene.end_frame_index, "scene_end")
            if config.sampling.frames_per_scene > 0:
                span = scene.end_frame_index - scene.start_frame_index
                for n in range(config.sampling.frames_per_scene):
                    fraction = (n + 1) / (config.sampling.frames_per_scene + 1)
                    add(round(scene.start_frame_index + span * fraction), "scene_representative")
    if mode == "all":
        if total > config.sampling.all_mode_safety_max_frames:
            raise ValueError(
                f"all sampling would process {total} frames, exceeding safety limit "
                f"{config.sampling.all_mode_safety_max_frames}"
            )
        for index in range(total):
            add(index, "all")
    if config.sampling.include_first_frame:
        add(0, "first")
    if config.sampling.include_last_frame and total > 0:
        add(total - 1, "last")
    return reasons


def sample_video(
    video: VideoRecord,
    output_dir: Path,
    config: AnalysisConfig,
) -> tuple[list[SceneRecord], list[FrameRecord]]:
    fps = video.average_fps or video.nominal_fps or 25.0
    scene_threshold = config.sampling.scene_threshold
    if scene_threshold is None:
        scene_threshold = 0.30 if config.sampling.scene_detector == "adaptive" else 0.40
    scenes = detect_scenes(str(video.path), video.video_id, threshold=scene_threshold)
    candidates = _candidate_indices(video, scenes, config)

    capture = cv2.VideoCapture(str(video.path))
    if not capture.isOpened():
        raise VideoDecodeError(f"Could not decode video: {video.path}")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    selected: list[FrameRecord] = []
    previous_decoded = None
    previous_saved_hash: str | None = None
    previous_saved_time: float | None = None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or video.frame_count or 0)
    adaptive = config.sampling.mode == "adaptive"
    index = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            timestamp = index / fps
            change = frame_change_score(previous_decoded, image)
            if (
                adaptive
                and config.sampling.motion_sampling
                and change >= config.sampling.motion_threshold
            ):
                candidates.setdefault(index, set()).add("high_motion")
            reasons = candidates.get(index)
            if reasons:
                if (
                    config.sampling.mode != "all"
                    and previous_saved_time is not None
                    and timestamp - previous_saved_time < config.sampling.minimum_gap_seconds
                    and not ({"first", "last", "scene_start", "scene_end"} & reasons)
                ):
                    previous_decoded = image
                    index += 1
                    continue
                current_hash = difference_hash(image)
                duplicate = False
                if (
                    config.sampling.mode != "all"
                    and config.sampling.deduplicate_frames
                    and previous_saved_hash is not None
                ):
                    duplicate = (
                        hamming_distance(previous_saved_hash, current_hash)
                        <= config.sampling.perceptual_hash_threshold
                    )
                if duplicate and not ({"first", "last", "scene_start", "scene_end"} & reasons):
                    previous_decoded = image
                    index += 1
                    continue
                frame_id = f"frame_{index:010d}"
                image_path = frames_dir / f"{frame_id}.jpg"
                if config.output.save_frames and not cv2.imwrite(str(image_path), image):
                    raise VideoDecodeError(f"Could not write sampled frame: {image_path}")
                record = FrameRecord(
                    video_id=video.video_id,
                    frame_id=frame_id,
                    source_frame_index=index,
                    timestamp_seconds=timestamp,
                    timestamp_ms=round(timestamp * 1000),
                    source_fps=fps,
                    scene_id=_scene_for_frame(index, scenes),
                    shot_id=_scene_for_frame(index, scenes),
                    sampling_reasons=sorted(reasons),
                    sampling_priority=float(len(reasons)) + change,
                    perceptual_hash=current_hash,
                    motion_score=change,
                    scene_change_score=change,
                    image_path=image_path if config.output.save_frames else None,
                    width=int(image.shape[1]),
                    height=int(image.shape[0]),
                    quality=analyze_frame_quality(image),
                )
                selected.append(record)
                previous_saved_hash = current_hash
                previous_saved_time = timestamp
            previous_decoded = image
            index += 1
            if (
                config.logging.process_progress_every > 0
                and index % config.logging.process_progress_every == 0
            ):
                LOGGER.info("Decoded %s/%s frames; selected %s", index, total or "?", len(selected))
    finally:
        capture.release()
    for pos, record in enumerate(selected):
        record.previous_sample_timestamp = selected[pos - 1].timestamp_seconds if pos > 0 else None
        record.next_sample_timestamp = (
            selected[pos + 1].timestamp_seconds if pos + 1 < len(selected) else None
        )
    return scenes, selected
