from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from legopolitics.exceptions import AudioExtractionError


def extract_audio(
    video_path: str | Path, output_path: str | Path, sample_rate: int = 16000
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise AudioExtractionError(
            "FFmpeg is required for audio extraction but was not found on PATH"
        )
    source = Path(video_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise AudioExtractionError(result.stderr.strip() or "FFmpeg audio extraction failed")
    return target
