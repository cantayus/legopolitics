import subprocess
import sys
from pathlib import Path

from legopolitics.constants import DISCLAIMER


def test_disclaimer_text():
    assert "not affiliated" in DISCLAIMER
    assert "does not sponsor" in DISCLAIMER


def test_release_gate_blocks_incomplete(tmp_path: Path):
    review = tmp_path / "review.md"
    review.write_text("completed: false", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts" / "check_trademark_review.py"
    result = subprocess.run(
        [sys.executable, str(script), str(review)], capture_output=True, text=True
    )
    assert result.returncode == 2
