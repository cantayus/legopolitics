from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from legopolitics.schemas.common import SchemaModel, utc_now


class RunProvenance(SchemaModel):
    run_id: str
    package_version: str
    git_commit: str | None = None
    git_dirty: bool | None = None
    python_version: str
    operating_system: str
    cpu: str
    ram_bytes: int | None = None
    gpu: str | None = None
    gpu_driver: str | None = None
    cuda_available: bool = False
    torch_version: str | None = None
    transformers_version: str | None = None
    dependency_versions: dict[str, str] = Field(default_factory=dict)
    configuration_hash: str
    codebook_hash: str | None = None
    random_seed: int
    device: str
    dtype: str
    remote_services_used: bool = False
    data_left_local_machine: bool = False
    started_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = None
    stage_durations: dict[str, float] = Field(default_factory=dict)
    peak_cpu_memory_bytes: int | None = None
    peak_gpu_memory_bytes: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
