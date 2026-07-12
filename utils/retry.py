from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    function: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 0.25,
    retry_for: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return function()
        except retry_for as exc:
            last = exc
            if attempt + 1 < attempts:
                time.sleep(base_delay * (2**attempt))
    assert last is not None
    raise last
