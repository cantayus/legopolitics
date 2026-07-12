from __future__ import annotations

from pathlib import Path

from legopolitics.config.models import AnalysisConfig
from legopolitics.exceptions import ConfigurationError, LicenseAcknowledgementError


def validate_runtime_config(config: AnalysisConfig, video_path: Path | None = None) -> None:
    if config.detection.enabled and config.detection.primary_backend == "yolo":
        if config.detection.yolo.weights is None:
            raise ConfigurationError("YOLO detection is enabled but no weights path was provided")
        if not config.legal.ultralytics_license_acknowledged:
            raise LicenseAcknowledgementError(
                "Ultralytics is enabled. Set legal.ultralytics_license_acknowledged=true "
                "after reviewing the applicable license terms."
            )
    candidate = video_path or config.input.video
    if candidate is not None and not Path(candidate).exists():
        raise ConfigurationError(f"Input video does not exist: {candidate}")
