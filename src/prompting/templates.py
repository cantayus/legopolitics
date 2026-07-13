from __future__ import annotations

BLIND_OBSERVATION_TEMPLATE = """Describe only what is visibly observable in the supplied image or sequence.
Do not infer nationality, political affiliation, religion, ethnicity, historical identity, intent,
morality, or causality unless supported by visible text, a recognizable symbol, or an explicitly
supplied identity reference. Separate clearly visible evidence, plausible but uncertain observations,
and details that cannot be determined. Return structured JSON."""

CONTEXTUAL_TEMPLATE = """Interpret the scene in light of the supplied context.
Maintain a strict distinction between direct visual evidence, textual evidence, audio evidence,
contextual inference, and uncertainty. Do not treat context as proof of ambiguous identity.
Provide evidence, confidence, contradictory evidence, and alternative explanations.

Event context: {event_context}
Research question: {research_question}
Known groups: {known_groups}
Inference constraints: {constraints}"""

CONTRADICTION_TEMPLATE = """Act as a skeptical second coder. Identify evidence that weakens,
contradicts, or fails to support the interpretation. Identify missing evidence and plausible
alternatives. Do not manufacture disagreement when evidence is clear.

Interpretation: {interpretation}"""

RELATION_TEMPLATE = """Extract subject-predicate-object relationships. Use only supplied track IDs.
Return start/end times, confidence, direct evidence IDs, and alternative predicates."""

NARRATIVE_TEMPLATE = """Aggregate frame-, track-, text-, and audio-level evidence into narrative
segments. Every conclusion must cite internal evidence identifiers and preserve uncertainty."""
