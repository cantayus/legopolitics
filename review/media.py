from pathlib import Path


def existing_media(path: str | None) -> Path | None:
    candidate = Path(path) if path else None
    return candidate if candidate and candidate.exists() else None
