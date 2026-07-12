import shutil
from pathlib import Path

from legopolitics.video import discover_videos


def test_deterministic_directory_batches(synthetic_video: Path, tmp_path: Path):
    source = tmp_path / "videos"
    source.mkdir()
    shutil.copy2(synthetic_video, source / "a.mp4")
    shutil.copy2(synthetic_video, source / "b.mp4")
    batch0 = discover_videos(source, True, num_batches=2, batch_id=0)
    batch1 = discover_videos(source, True, num_batches=2, batch_id=1)
    assert set(batch0).isdisjoint(batch1)
    assert set(batch0) | set(batch1)
