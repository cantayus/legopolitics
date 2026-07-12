from __future__ import annotations

from typing import Any

from legopolitics.exceptions import DependencyUnavailableError


class Sam3Segmenter:
    """Future-compatible gate for a SAM 3 adapter supplied through the plugin system."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DependencyUnavailableError(
            "No stable built-in SAM 3 API is assumed. Install a compatible external plugin "
            "registered under `legopolitics.segmenters`."
        )
