from pathlib import Path

import pytest

from legopolitics.exceptions import ConfigurationError
from legopolitics.ocr.provider import ProviderOCRAdapter


def test_provider_ocr_adapter_maps_structured_results(tmp_path: Path) -> None:
    image = tmp_path / "frame.png"
    image.write_bytes(b"test")
    adapter = ProviderOCRAdapter(
        lambda _: [
            {
                "text": "  Victory now  ",
                "confidence": 0.9,
                "language": "en",
                "box": [1, 2, 30, 40],
            }
        ],
        "test-provider",
    )
    records = adapter.recognize(image, "frame_1", "video_1")
    assert len(records) == 1
    assert records[0].normalized_text == "Victory now"
    assert records[0].bounding_box is not None
    assert records[0].bounding_box.x2 == 30


def test_remote_provider_ocr_requires_opt_in() -> None:
    with pytest.raises(ConfigurationError):
        ProviderOCRAdapter(lambda _: [], "remote", is_remote=True)
