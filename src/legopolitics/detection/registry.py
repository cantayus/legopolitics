from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from legopolitics.exceptions import ConfigurationError


class AdapterRegistry:
    def __init__(self, group: str) -> None:
        self.group = group
        self._factories: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, factory: Callable[..., Any]) -> None:
        if name in self._factories:
            raise ConfigurationError(f"Adapter already registered: {name}")
        self._factories[name] = factory

    def load_entry_points(self) -> None:
        for item in entry_points().select(group=self.group):
            self._factories.setdefault(item.name, item.load())

    def create(self, name: str, **kwargs: Any) -> Any:
        if name not in self._factories:
            self.load_entry_points()
        try:
            return self._factories[name](**kwargs)
        except KeyError as exc:
            raise ConfigurationError(f"Unknown adapter '{name}' in registry {self.group}") from exc

    def names(self) -> list[str]:
        self.load_entry_points()
        return sorted(self._factories)


DETECTORS = AdapterRegistry("legopolitics.detectors")
