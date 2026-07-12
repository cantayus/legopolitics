from __future__ import annotations


def contradiction_prompt(interpretation: str) -> str:
    return (
        "Identify evidence that weakens, contradicts, or fails to support the proposed interpretation. "
        "Identify missing evidence and plausible alternatives. Do not manufacture disagreement.\n\n"
        f"Interpretation:\n{interpretation}"
    )
