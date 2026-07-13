from __future__ import annotations

from legopolitics.utils.fingerprints import stable_hash

PROMPT_VERSION = "1.0"


def prompt_hash(prompt: str) -> str:
    return stable_hash({"version": PROMPT_VERSION, "prompt": prompt})
