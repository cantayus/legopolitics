from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from legopolitics.config.models import AnalysisConfig
from legopolitics.exceptions import ConfigurationError

CONFIG_PROFILES = {
    "default": None,
    "lego-yolo": "lego_figure_detector.yaml",
    "hf-ensemble": "huggingface_vlm_ensemble.yaml",
    "hf-cpu": "huggingface_vlm_cpu.yaml",
    "hf-small-gpu": "huggingface_vlm_small_gpu.yaml",
    "hf-midrange-gpu": "huggingface_vlm_midrange_gpu.yaml",
    "hf-high-end-gpu": "huggingface_vlm_high_end_gpu.yaml",
    "hf-maximum": "huggingface_vlm_maximum.yaml",
}


def write_default_config(path: str | Path, profile: str = "default") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if profile not in CONFIG_PROFILES:
        choices = ", ".join(sorted(CONFIG_PROFILES))
        raise ConfigurationError(f"Unknown configuration profile {profile!r}. Choose: {choices}")
    resource_name = CONFIG_PROFILES[profile]
    if resource_name is None:
        return AnalysisConfig().to_yaml(target)
    resource = files("legopolitics.resources").joinpath(resource_name)
    target.write_text(resource.read_text(encoding="utf-8"), encoding="utf-8")
    return target
