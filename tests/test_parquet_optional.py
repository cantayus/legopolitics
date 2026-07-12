import importlib.util

import pytest

from legopolitics.exceptions import DependencyUnavailableError
from legopolitics.storage.parquet import write_parquet


def test_optional_parquet_behavior(tmp_path):
    path = write_parquet(tmp_path / "x.parquet", [{"a": 1}], strict=False)
    if importlib.util.find_spec("pyarrow"):
        assert path and path.exists()
    else:
        assert path is None
        with pytest.raises(DependencyUnavailableError):
            write_parquet(tmp_path / "x.parquet", [{"a": 1}], strict=True)


def test_parquet_frame_normalizes_mixed_object_values():
    from legopolitics.storage.parquet import parquet_frame

    frame = parquet_frame(
        [
            {"name": "count", "value": 2},
            {"name": "label", "value": "unknown"},
            {"name": "metadata", "value": {"visible": True}},
        ]
    )
    assert str(frame["value"].dtype) == "string"
    assert frame["value"].tolist() == ["2", "unknown", '{"visible": true}']
