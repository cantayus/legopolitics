from __future__ import annotations

from typing import Any

from legopolitics.exceptions import DependencyUnavailableError


class Sam3TrackAdapter:
    """Plugin gate for future or third-party SAM 3 tracking implementations.

    The package does not guess an unstable API. External implementations should
    register under the ``legopolitics.trackers`` entry-point group. Constructing
    this built-in gate produces an actionable error instead of silently falling
    back to a different tracking algorithm.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DependencyUnavailableError(
            "No stable built-in SAM 3 tracking API is assumed. Install a compatible "
            "external plugin registered under `legopolitics.trackers`."
        )
