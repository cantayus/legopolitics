from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from legopolitics.utils.device import detect_device


def dependency_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def git_state(cwd: Path | None = None) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def total_ram_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        if hasattr(os, "sysconf"):
            try:
                return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
            except (ValueError, OSError):
                return None
        return None


def inspect_environment() -> dict[str, Any]:
    device = detect_device()
    commit, dirty = git_state()
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "cpu": platform.processor() or platform.machine(),
        "ram_bytes": total_ram_bytes(),
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "device": device.device,
        "cuda_available": device.cuda_available,
        "gpu_name": device.gpu_name,
        "gpu_memory_bytes": device.total_gpu_memory_bytes,
        "torch_version": device.torch_version,
        "git_commit": commit,
        "git_dirty": dirty,
        "dependencies": dependency_versions(
            [
                "pydantic",
                "typer",
                "PyYAML",
                "numpy",
                "pandas",
                "Pillow",
                "opencv-python-headless",
                "openpyxl",
                "jinja2",
                "rich",
                "pyarrow",
                "ultralytics",
                "torch",
                "transformers",
                "faster-whisper",
                "pyannote.audio",
                "streamlit",
            ]
        ),
        "tokens_present": {
            "huggingface": bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
        },
    }
