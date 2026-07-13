from __future__ import annotations

from pathlib import Path

from legopolitics.schemas import AnalysisResult


def load_completed_result(
    path: Path, configuration_hash: str, video_fingerprint: str
) -> AnalysisResult | None:
    if not path.exists():
        return None
    try:
        result = AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if result.video.fingerprint != video_fingerprint:
        return None
    if result.provenance is None or result.provenance.configuration_hash != configuration_hash:
        return None
    return result
