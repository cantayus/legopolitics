from __future__ import annotations

from legopolitics.schemas import OCRRecord, TrackObservation


def associate_ocr_with_visible_tracks(
    ocr: list[OCRRecord], observations: list[TrackObservation]
) -> None:
    by_frame: dict[str, list[str]] = {}
    for item in observations:
        by_frame.setdefault(item.frame_id, []).append(item.track_id)
    for record in ocr:
        candidates = by_frame.get(record.frame_id, [])
        if len(candidates) == 1:
            record.track_id = candidates[0]
