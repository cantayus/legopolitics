from pathlib import Path

from openpyxl import load_workbook

from legopolitics import AnalysisConfig, LegoPoliticsAnalyzer
from legopolitics.storage import load_result


def test_end_to_end_and_resume(synthetic_video: Path, tmp_path: Path):
    config = AnalysisConfig().with_overrides(
        **{
            "sampling.mode": "hybrid",
            "sampling.fps": 2.0,
            "detection.enabled": True,
            "detection.primary_backend": "mock",
            "segmentation.enabled": True,
            "segmentation.backend": "box",
            "output.parquet": False,
        }
    )
    output = tmp_path / "result"
    result = LegoPoliticsAnalyzer(config).analyze_video(synthetic_video, output)
    assert result.frames and result.detections and result.tracks and result.masks
    assert (output / "video_summary.xlsx").exists()
    assert (output / "report.html").exists()
    assert (output / "run_manifest.sqlite").exists()
    assert (output / "run_config.yaml").exists()
    assert (output / "effective_config.yaml").exists()
    assert (output / "environment.json").exists()
    assert (output / "provenance.json").exists()
    assert (output / "jsonl" / "parsed_outputs.jsonl").exists()
    assert (output / "jsonl" / "raw_model_outputs.jsonl").exists()
    assert (output / "jsonl" / "rendered_prompts.jsonl").exists()
    load_workbook(output / "video_summary.xlsx", read_only=True).close()
    loaded = load_result(output / "result.json")
    assert loaded.video.fingerprint == result.video.fingerprint
    resumed = LegoPoliticsAnalyzer(config).analyze_video(synthetic_video, output)
    assert resumed.provenance.configuration_hash == result.provenance.configuration_hash
