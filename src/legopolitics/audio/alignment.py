from __future__ import annotations

from legopolitics.schemas import DiarizationSegment, FrameRecord, TranscriptSegment


def align_speakers(
    transcript: list[TranscriptSegment], diarization: list[DiarizationSegment]
) -> None:
    for segment in transcript:
        overlaps = []
        for diar in diarization:
            overlap = max(
                0.0,
                min(segment.end_timestamp, diar.end_timestamp)
                - max(segment.start_timestamp, diar.start_timestamp),
            )
            if overlap > 0:
                overlaps.append((overlap, diar.speaker_label))
        if overlaps:
            segment.speaker_label = max(overlaps)[1]


def frames_for_interval(frames: list[FrameRecord], start: float, end: float) -> list[str]:
    return [f.frame_id for f in frames if start <= f.timestamp_seconds <= end]
