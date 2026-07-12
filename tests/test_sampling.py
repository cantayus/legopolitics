from pathlib import Path

from legopolitics import AnalysisConfig
from legopolitics.video import probe_video, sample_video


def run_mode(synthetic_video: Path, tmp_path: Path, mode: str):
    config = AnalysisConfig().with_overrides(
        **{
            "sampling.mode": mode,
            "sampling.fps": 2.0,
            "output.save_frames": True,
            "output.parquet": False,
        }
    )
    return sample_video(probe_video(synthetic_video), tmp_path / mode, config)


def test_fixed_sampling(synthetic_video: Path, tmp_path: Path):
    scenes, frames = run_mode(synthetic_video, tmp_path, "fixed")
    assert scenes
    assert 2 <= len(frames) <= 12
    assert all(frame.image_path and frame.image_path.exists() for frame in frames)


def test_scene_hybrid_adaptive_all(synthetic_video: Path, tmp_path: Path):
    for mode in ["scene", "hybrid", "adaptive", "all"]:
        scenes, frames = run_mode(synthetic_video, tmp_path, mode)
        assert scenes and frames
    _, all_frames = run_mode(synthetic_video, tmp_path / "second", "all")
    assert len(all_frames) >= 30
