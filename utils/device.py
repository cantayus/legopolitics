from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

HardwareProfile = Literal["auto", "cpu", "small_gpu", "midrange_gpu", "high_end_gpu", "maximum"]


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    cuda_available: bool
    gpu_name: str | None
    total_gpu_memory_bytes: int | None
    torch_version: str | None
    bf16_supported: bool = False

    @property
    def total_gpu_memory_gb(self) -> float | None:
        if self.total_gpu_memory_bytes is None:
            return None
        return self.total_gpu_memory_bytes / (1024**3)


@dataclass(frozen=True)
class VLMRuntimePlan:
    profile: str
    device: str
    dtype: str
    quantization: str
    device_map: str | dict[str, int | str] | None
    max_memory: dict[str, int | str]
    offload_folder: str | None


def detect_device(requested: str = "auto") -> DeviceInfo:
    try:
        import torch
    except ImportError:
        return DeviceInfo("cpu", False, None, None, None, False)
    cuda = bool(torch.cuda.is_available())
    if requested not in {"auto", "cpu"}:
        device = requested
    else:
        device = "cuda:0" if requested == "auto" and cuda else "cpu"
    name = torch.cuda.get_device_name(0) if cuda else None
    memory = torch.cuda.get_device_properties(0).total_memory if cuda else None
    bf16 = bool(torch.cuda.is_bf16_supported()) if cuda else False
    return DeviceInfo(device, cuda, name, memory, torch.__version__, bf16)


def recommend_hardware_profile(info: DeviceInfo | None = None) -> str:
    """Recommend a conservative VLM profile from available GPU memory."""
    info = info or detect_device("auto")
    if not info.cuda_available or info.total_gpu_memory_gb is None:
        return "cpu"
    memory = info.total_gpu_memory_gb
    if memory < 6:
        return "small_gpu"
    if memory < 14:
        return "midrange_gpu"
    if memory < 28:
        return "high_end_gpu"
    return "maximum"


def resolve_vlm_runtime_plan(
    *,
    requested_device: str = "auto",
    requested_dtype: str = "auto",
    requested_profile: HardwareProfile = "auto",
    quantization: str = "none",
    device_map: str | dict[str, int | str] | None = None,
    max_memory: dict[str, int | str] | None = None,
    offload_folder: str | None = None,
) -> VLMRuntimePlan:
    """Resolve portable VLM loading settings without loading a model.

    Explicit per-model settings always win. The profile controls conservative
    defaults only, so researchers can reproduce a run across different machines.
    """
    info = detect_device(requested_device)
    profile = recommend_hardware_profile(info) if requested_profile == "auto" else requested_profile
    device = info.device
    if profile == "cpu":
        device = "cpu"

    if requested_dtype != "auto":
        dtype = requested_dtype
    elif device == "cpu":
        dtype = "float32"
    elif info.bf16_supported and profile in {"high_end_gpu", "maximum"}:
        dtype = "bfloat16"
    else:
        dtype = "float16"

    resolved_device_map = device_map
    if resolved_device_map is None and quantization != "none":
        resolved_device_map = "auto"
    if resolved_device_map is None and profile in {"small_gpu", "midrange_gpu"}:
        # Allows Accelerate to offload layers when a selected model is larger than VRAM.
        resolved_device_map = "auto"

    return VLMRuntimePlan(
        profile=profile,
        device=device,
        dtype=dtype,
        quantization=quantization,
        device_map=resolved_device_map,
        max_memory=dict(max_memory or {}),
        offload_folder=offload_folder,
    )


def resolve_torch_dtype(name: str, device: str) -> Any:
    try:
        import torch
    except ImportError:
        return None
    normalized = name.lower()
    if normalized == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    if device == "cpu" and mapping[normalized] == torch.float16:
        return torch.float32
    return mapping[normalized]
