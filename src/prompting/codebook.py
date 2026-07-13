from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from legopolitics.exceptions import ConfigurationError


class CodebookEntry(BaseModel):
    variable_id: str
    label: str
    description: str
    unit_of_analysis: str = "frame"
    allowed_values: list[Any] = Field(default_factory=list)
    scale: dict[str, float] | None = None
    inclusion_criteria: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    positive_examples: list[str] = Field(default_factory=list)
    negative_examples: list[str] = Field(default_factory=list)
    ambiguous_examples: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    prompt_instructions: str = ""
    aggregation_method: str = "mean"


class Codebook(BaseModel):
    schema_version: str = "1.0"
    entries: list[CodebookEntry] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Codebook:
        try:
            return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
        except (OSError, yaml.YAMLError, ValueError) as exc:
            raise ConfigurationError(f"Could not load codebook: {exc}") from exc
