from __future__ import annotations

from typing import Any


def memory_snapshot() -> dict[str, Any]:
    """Collect available process and CUDA memory counters without hard dependencies."""
    result: dict[str, Any] = {}
    try:
        import psutil

        process = psutil.Process()
        result["psutil_available"] = True
        result["rss_bytes"] = int(process.memory_info().rss)
    except ImportError:
        result["psutil_available"] = False
    try:
        import torch

        result["torch_available"] = True
        if torch.cuda.is_available():
            result["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated())
            result["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
            result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    except ImportError:
        result["torch_available"] = False
    return result


def release_cuda_cache() -> None:
    """Release PyTorch's unused CUDA cache when PyTorch and CUDA are available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return
