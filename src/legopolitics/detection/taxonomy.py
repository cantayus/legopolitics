from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaxonomyNode:
    name: str
    parent: str | None = None
    aliases: list[str] = field(default_factory=list)
    multilingual_labels: dict[str, str] = field(default_factory=dict)


DEFAULT_TAXONOMY = [
    TaxonomyNode("brick-built figure", aliases=["brick figure", "minifigure"]),
    TaxonomyNode("soldier-like figure", parent="brick-built figure"),
    TaxonomyNode("civilian-like figure", parent="brick-built figure"),
    TaxonomyNode("leader-like figure", parent="brick-built figure"),
    TaxonomyNode("child-like figure", parent="brick-built figure"),
    TaxonomyNode("captured figure", parent="brick-built figure"),
    TaxonomyNode("prone figure", parent="brick-built figure"),
    TaxonomyNode("rifle", parent="weapon"),
    TaxonomyNode("handgun", parent="weapon"),
    TaxonomyNode("missile", parent="military object"),
    TaxonomyNode("drone", parent="military object"),
    TaxonomyNode("tank", parent="military vehicle"),
    TaxonomyNode("flag", parent="political symbol"),
    TaxonomyNode("national emblem", parent="political symbol"),
    TaxonomyNode("hospital", parent="civilian object"),
    TaxonomyNode("ambulance", parent="civilian vehicle"),
    TaxonomyNode("subtitle", parent="media element"),
    TaxonomyNode("speech bubble", parent="media element"),
]
