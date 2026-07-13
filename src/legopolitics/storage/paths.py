from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from legopolitics.schemas import OutputPaths


@dataclass(frozen=True)
class OutputLayout:
    root: Path
    tables: Path
    jsonl: Path
    frames: Path
    annotated_frames: Path
    crops: Path
    track_contact_sheets: Path
    scene_contact_sheets: Path
    masks: Path
    audio: Path
    validation: Path
    cache: Path
    logs: Path

    @classmethod
    def create(cls, root: str | Path) -> OutputLayout:
        base = Path(root)
        layout = cls(
            base,
            base / "tables",
            base / "jsonl",
            base / "frames",
            base / "annotated_frames",
            base / "crops",
            base / "track_contact_sheets",
            base / "scene_contact_sheets",
            base / "masks",
            base / "audio",
            base / "validation",
            base / "cache",
            base / "logs",
        )
        for path in layout.__dict__.values():
            Path(path).mkdir(parents=True, exist_ok=True)
        return layout

    def output_paths(self) -> OutputPaths:
        return OutputPaths(
            root=self.root,
            workbook=self.root / "video_summary.xlsx",
            report=self.root / "report.html",
            manifest=self.root / "run_manifest.sqlite",
            tables_dir=self.tables,
            jsonl_dir=self.jsonl,
            frames_dir=self.frames,
            crops_dir=self.crops,
            masks_dir=self.masks,
            audio_dir=self.audio,
        )
