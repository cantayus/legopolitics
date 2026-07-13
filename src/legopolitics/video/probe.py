from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

from legopolitics.exceptions import VideoProbeError
from legopolitics.schemas import VideoRecord
from legopolitics.utils.fingerprints import fast_file_fingerprint
from legopolitics.video.discovery import normalize_path


def _fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _probe_ffprobe(path: Path) -> dict[str, Any] | None:
    if shutil.which("ffprobe") is None:
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return json.loads(output)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return None


def probe_video(path: str | Path) -> VideoRecord:
    source = Path(path)
    if not source.exists():
        raise VideoProbeError(f"Video does not exist: {source}")
    stat = source.stat()
    fingerprint = fast_file_fingerprint(source)
    video_id = f"video_{uuid.uuid5(uuid.NAMESPACE_URL, fingerprint).hex[:16]}"
    payload = _probe_ffprobe(source)
    if payload:
        streams = payload.get("streams", [])
        video_streams = [s for s in streams if s.get("codec_type") == "video"]
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
        stream = video_streams[0] if video_streams else {}
        fmt = payload.get("format", {})
        avg_fps = _fraction(stream.get("avg_frame_rate"))
        nominal_fps = _fraction(stream.get("r_frame_rate"))
        duration = stream.get("duration") or fmt.get("duration")
        count = stream.get("nb_frames")
        tags = stream.get("tags", {})
        rotation = stream.get("rotation") or tags.get("rotate")
        return VideoRecord(
            video_id=video_id,
            path=source,
            normalized_path=normalize_path(source),
            fingerprint=fingerprint,
            file_size=stat.st_size,
            modified_ns=stat.st_mtime_ns,
            duration_seconds=float(duration) if duration else None,
            frame_count=int(count) if count and str(count).isdigit() else None,
            average_fps=avg_fps,
            nominal_fps=nominal_fps,
            variable_frame_rate=bool(avg_fps and nominal_fps and abs(avg_fps - nominal_fps) > 0.01),
            width=int(stream["width"]) if stream.get("width") else None,
            height=int(stream["height"]) if stream.get("height") else None,
            rotation=int(rotation) if rotation not in (None, "") else None,
            pixel_format=stream.get("pix_fmt"),
            codec=stream.get("codec_name"),
            bit_rate=int(stream.get("bit_rate") or fmt.get("bit_rate"))
            if (stream.get("bit_rate") or fmt.get("bit_rate"))
            else None,
            time_base=stream.get("time_base"),
            has_audio=bool(audio_streams),
            audio_streams=len(audio_streams),
            subtitle_streams=len(subtitle_streams),
            metadata={"ffprobe": payload},
        )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoProbeError(f"Could not open video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else None
    finally:
        capture.release()
    return VideoRecord(
        video_id=video_id,
        path=source,
        normalized_path=normalize_path(source),
        fingerprint=fingerprint,
        file_size=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        duration_seconds=duration,
        frame_count=frame_count or None,
        average_fps=fps or None,
        nominal_fps=fps or None,
        variable_frame_rate=None,
        width=width or None,
        height=height or None,
        codec=None,
        has_audio=False,
        metadata={"probe_backend": "opencv"},
    )
