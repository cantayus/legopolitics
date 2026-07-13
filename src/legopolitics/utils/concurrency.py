from __future__ import annotations

import os
import socket
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerIdentity:
    hostname: str
    process_id: int

    @property
    def value(self) -> str:
        return f"{self.hostname}:{self.process_id}"


def current_worker() -> WorkerIdentity:
    return WorkerIdentity(socket.gethostname(), os.getpid())
