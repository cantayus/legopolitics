from __future__ import annotations

import logging
from dataclasses import dataclass, field

from legopolitics.config import AnalysisConfig
from legopolitics.schemas import ErrorRecord
from legopolitics.storage import OutputLayout, RunManifest


@dataclass
class PipelineContext:
    config: AnalysisConfig
    layout: OutputLayout
    manifest: RunManifest
    run_id: str
    logger: logging.Logger
    errors: list[ErrorRecord] = field(default_factory=list)
