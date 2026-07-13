from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

T = TypeVar("T")


def progress(
    iterable: Iterable[T], description: str, enabled: bool = True, total: int | None = None
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return
    try:
        from rich.progress import track

        yield from track(iterable, description=description, total=total)
    except ImportError:
        yield from iterable
