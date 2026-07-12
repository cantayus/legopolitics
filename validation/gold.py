from typing import TypeVar

T = TypeVar("T")


def resolved_label(automated: T, correction: T | None) -> T:
    return correction if correction is not None else automated
