from __future__ import annotations

from pathlib import Path
from typing import Any

from legopolitics.schemas import AnalysisResult
from legopolitics.storage.atomic import atomic_write_text
from legopolitics.storage.excel import write_workbook
from legopolitics.storage.jsonl import write_jsonl
from legopolitics.storage.parquet import write_parquet
from legopolitics.storage.sqlite import write_table

TABLE_MAP = {
    "Video": "video",
    "Scenes": "scenes",
    "Frames": "frames",
    "Detections": "detections",
    "Fused_Detections": "fused_detections",
    "Tracks": "tracks",
    "Track_Observations": "track_observations",
    "Attributes": "attributes",
    "Relations": "relations",
    "OCR": "ocr",
    "Transcript": "transcript",
    "Diarization": "diarization",
    "Audio_Events": "audio_events",
    "Events": "events",
    "Narrative_Segments": "narrative_segments",
    "Codebook_Scores": "codebook_scores",
    "Model_Agreement": "model_agreement",
    "Validation": "validation_summary",
    "Errors": "errors",
    "Prompts": "visual_responses",
}


def result_tables(result: AnalysisResult) -> dict[str, list[Any]]:
    tables = {}
    for sheet, attribute in TABLE_MAP.items():
        value = getattr(result, attribute)
        tables[sheet] = [value] if attribute == "video" else list(value)
    model_rows: dict[tuple[str, str | None, str | None], dict[str, Any]] = {}
    for response in result.visual_responses:
        provenance = response.provenance
        key = (provenance.adapter, provenance.model_id, provenance.revision)
        model_rows[key] = provenance.model_dump(mode="json")
    for detection in result.detections:
        key = (detection.detector_id, detection.model_id, detection.model_revision)
        model_rows.setdefault(
            key,
            {
                "adapter": detection.detector_id,
                "model_id": detection.model_id,
                "revision": detection.model_revision,
                "weight_hash": detection.weight_hash,
            },
        )
    tables["Models"] = list(model_rows.values())
    return tables


def export_result(result: AnalysisResult, config: Any) -> dict[str, Path]:
    root = result.output_paths.root
    tables_dir = result.output_paths.tables_dir or root / "tables"
    jsonl_dir = result.output_paths.jsonl_dir or root / "jsonl"
    database = result.output_paths.manifest or root / "run_manifest.sqlite"
    tables = result_tables(result)
    artifacts = {}
    if config.output.jsonl:
        for sheet, records in tables.items():
            artifacts[f"jsonl_{sheet}"] = write_jsonl(jsonl_dir / f"{sheet.lower()}.jsonl", records)
        artifacts["parsed_outputs"] = write_jsonl(
            jsonl_dir / "parsed_outputs.jsonl", result.visual_responses
        )
        artifacts["errors_jsonl"] = write_jsonl(jsonl_dir / "errors.jsonl", result.errors)
        artifacts["raw_model_outputs"] = (
            write_jsonl(jsonl_dir / "raw_model_outputs.jsonl", [])
            if not (jsonl_dir / "raw_model_outputs.jsonl").exists()
            else jsonl_dir / "raw_model_outputs.jsonl"
        )
        artifacts["rendered_prompts"] = (
            write_jsonl(jsonl_dir / "rendered_prompts.jsonl", [])
            if not (jsonl_dir / "rendered_prompts.jsonl").exists()
            else jsonl_dir / "rendered_prompts.jsonl"
        )
        artifacts["result_json"] = atomic_write_text(
            root / "result.json", result.model_dump_json(indent=2)
        )
    if config.output.sqlite:
        for sheet, records in tables.items():
            write_table(database, "data_" + sheet.lower(), records)
        artifacts["sqlite"] = database
    if config.output.parquet:
        for sheet, records in tables.items():
            path = write_parquet(
                tables_dir / f"{sheet.lower()}.parquet",
                records,
                strict=config.output.strict_parquet,
            )
            if path:
                artifacts[f"parquet_{sheet}"] = path
    if config.output.excel:
        metadata = result.provenance.model_dump(mode="json") if result.provenance else {}
        artifacts["excel"] = write_workbook(
            result.output_paths.workbook or root / "video_summary.xlsx",
            tables,
            metadata,
            config.output.include_excel_charts,
        )
    return artifacts


def load_result(path: Path) -> AnalysisResult:
    return AnalysisResult.model_validate_json(path.read_text(encoding="utf-8"))
