from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class PipelineEvent:
    stage: str
    status: str
    message: str
    timestamp: datetime = datetime.now(timezone.utc)
