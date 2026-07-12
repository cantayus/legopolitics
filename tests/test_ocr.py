from legopolitics.ocr import deduplicate_temporally
from legopolitics.schemas import OCRRecord


def test_temporal_ocr_deduplication():
    records = [
        OCRRecord(
            text_region_id="1",
            video_id="v",
            frame_id="f1",
            original_text="Victory now",
            normalized_text="Victory now",
            backend="mock",
        ),
        OCRRecord(
            text_region_id="2",
            video_id="v",
            frame_id="f2",
            original_text="Victory now!",
            normalized_text="Victory now!",
            backend="mock",
        ),
    ]
    result = deduplicate_temporally(records, 0.85)
    assert len(result) == 1
    assert result[0].repetition_count == 2
