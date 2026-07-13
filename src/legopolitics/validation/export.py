from pathlib import Path

from legopolitics.storage.jsonl import write_jsonl


def export_validation(path: Path, records: list) -> Path:
    return write_jsonl(path, records)
