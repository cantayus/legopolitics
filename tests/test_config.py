from pathlib import Path

import pytest

from legopolitics import AnalysisConfig


def test_round_trip_yaml(tmp_path: Path):
    config = AnalysisConfig().with_overrides(**{"sampling.fps": 2.0, "output.parquet": False})
    path = config.to_yaml(tmp_path / "config.yaml")
    loaded = AnalysisConfig.from_yaml(path)
    assert loaded.sampling.fps == 2.0
    assert loaded.config_hash() == config.config_hash()


def test_invalid_sampling():
    with pytest.raises(ValueError):
        AnalysisConfig().with_overrides(**{"sampling.fps": 0})


def test_unknown_fields_forbidden():
    with pytest.raises(ValueError):
        AnalysisConfig.model_validate({"unknown": True})


def test_multiple_huggingface_vlms_can_be_configured():
    config = AnalysisConfig.model_validate(
        {
            "vision": {
                "models": {
                    "huggingface_vlms": [
                        {
                            "name": "smolvlm2",
                            "enabled": True,
                            "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                            "input_mode": "chat",
                        },
                        {
                            "name": "qwen",
                            "enabled": True,
                            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                            "processor_kwargs": {"max_pixels": 802816},
                        },
                    ]
                }
            }
        }
    )
    assert len(config.vision.models.huggingface_vlms) == 2
    assert config.vision.models.huggingface_vlms[1].processor_kwargs["max_pixels"] == 802816
