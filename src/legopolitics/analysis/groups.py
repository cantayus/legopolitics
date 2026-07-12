from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GroupAssignment:
    assigned_group: str | None
    confidence: float
    assignment_method: str
    supporting_evidence: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)
    alternative_groups: list[str] = field(default_factory=list)
    manual_override: bool = False
