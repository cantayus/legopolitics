from pathlib import Path

from legopolitics.video import discover_videos, probe_video


def test_discover_and_probe(synthetic_video: Path):
    videos = discover_videos(synthetic_video.parent, recursive=True)
    assert synthetic_video in videos
    record = probe_video(synthetic_video)
    assert record.width == 160
    assert record.height == 120
    assert record.frame_count in {40, None} or record.frame_count > 0
    assert record.fingerprint
