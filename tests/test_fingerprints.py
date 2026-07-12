from pathlib import Path

from legopolitics.utils.fingerprints import (
    deterministic_partition,
    fast_file_fingerprint,
    stable_hash,
)


def test_fingerprint_stable(tmp_path: Path):
    path = tmp_path / "a.bin"
    path.write_bytes(b"abc")
    assert fast_file_fingerprint(path) == fast_file_fingerprint(path)
    assert stable_hash({"a": 1}) == stable_hash({"a": 1})


def test_partition_range():
    values = [deterministic_partition(f"video-{i}", 4) for i in range(20)]
    assert all(0 <= value < 4 for value in values)
