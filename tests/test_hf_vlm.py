from legopolitics import AnalysisConfig, LegoPoliticsAnalyzer
from legopolitics.utils.device import (
    DeviceInfo,
    recommend_hardware_profile,
    resolve_vlm_runtime_plan,
)
from legopolitics.vision.hf_vlm import HuggingFaceVLMAdapter


def test_chat_generated_text_extraction():
    payload = [
        {
            "generated_text": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
    ]
    assert HuggingFaceVLMAdapter._generated_text(payload) == "answer"


def test_analyzer_builds_multiple_hf_adapters_without_loading_models():
    config = AnalysisConfig.model_validate(
        {
            "vision": {
                "models": {
                    "huggingface_vlms": [
                        {
                            "name": "smolvlm2",
                            "enabled": True,
                            "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                        },
                        {
                            "name": "qwen",
                            "enabled": True,
                            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                        },
                    ]
                }
            }
        }
    )
    adapters = LegoPoliticsAnalyzer(config)._vision_adapters()
    assert [adapter.metadata().adapter for adapter in adapters] == ["smolvlm2", "qwen"]


def test_hardware_profile_recommendations_cover_cpu_to_large_gpu():
    assert recommend_hardware_profile(DeviceInfo("cpu", False, None, None, "2.6")) == "cpu"
    assert (
        recommend_hardware_profile(DeviceInfo("cuda:0", True, "small", 4 * 1024**3, "2.6"))
        == "small_gpu"
    )
    assert (
        recommend_hardware_profile(DeviceInfo("cuda:0", True, "mid", 8 * 1024**3, "2.6"))
        == "midrange_gpu"
    )
    assert (
        recommend_hardware_profile(DeviceInfo("cuda:0", True, "high", 16 * 1024**3, "2.6"))
        == "high_end_gpu"
    )
    assert (
        recommend_hardware_profile(DeviceInfo("cuda:0", True, "large", 48 * 1024**3, "2.6"))
        == "maximum"
    )


def test_runtime_plan_preserves_explicit_quantization_and_offload():
    plan = resolve_vlm_runtime_plan(
        requested_device="cpu",
        requested_dtype="float32",
        requested_profile="cpu",
        quantization="none",
        device_map=None,
        max_memory={"cpu": "32GiB"},
        offload_folder="cache/offload",
    )
    assert plan.profile == "cpu"
    assert plan.device == "cpu"
    assert plan.dtype == "float32"
    assert plan.max_memory == {"cpu": "32GiB"}
    assert plan.offload_folder == "cache/offload"


def test_model_config_accepts_quantized_hf_settings():
    config = AnalysisConfig.model_validate(
        {
            "performance": {"hardware_profile": "midrange_gpu"},
            "vision": {
                "models": {
                    "huggingface_vlms": [
                        {
                            "name": "qwen",
                            "enabled": True,
                            "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                            "quantization": "4bit",
                            "device_map": "auto",
                            "max_memory": {"0": "7GiB", "cpu": "32GiB"},
                            "offload_folder": "cache/offload",
                        }
                    ]
                }
            },
        }
    )
    model = config.vision.models.huggingface_vlms[0]
    assert model.quantization == "4bit"
    assert model.device_map == "auto"
    assert model.max_memory["0"] == "7GiB"
