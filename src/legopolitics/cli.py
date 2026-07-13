from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from legopolitics import AnalysisConfig, LegoPoliticsAnalyzer, __version__
from legopolitics.config import write_default_config
from legopolitics.constants import DISCLAIMER
from legopolitics.detection import inspect_yolo_weights, register_yolo_weights
from legopolitics.storage import RunManifest, export_result, load_result
from legopolitics.storage.migrations import migrate_output
from legopolitics.utils.device import detect_device, recommend_hardware_profile
from legopolitics.utils.environment import inspect_environment

app = typer.Typer(
    help="Multimodal research analysis of political brick-built videos.",
    no_args_is_help=True,
)
console = Console()


def _config(path: Path | None) -> AnalysisConfig:
    return AnalysisConfig.from_yaml(path) if path else AnalysisConfig()


@app.command("analyze-video")
def analyze_video(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the input video file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
            help="Directory where analysis outputs will be written.",
        ),
    ],
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Optional YAML configuration file.",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Resume an existing run when possible.",
        ),
    ] = True,
) -> None:
    cfg = _config(config).with_overrides(**{"project.resume": resume})
    result = LegoPoliticsAnalyzer(cfg).analyze_video(video, output)

    console.print(
        {
            "video_id": result.video.video_id,
            "frames": len(result.frames),
            "detections": len(result.detections),
            "tracks": len(result.tracks),
            "output": str(result.output_paths.root),
        }
    )


@app.command("analyze-directory")
def analyze_directory(
    input_root: Annotated[
        Path,
        typer.Option(
            "--input-root",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option(
            "--output-root",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
    recursive: bool = True,
    num_batches: int = 1,
    batch_id: int = 0,
    resume: bool = True,
) -> None:
    results = LegoPoliticsAnalyzer(_config(config)).analyze_directory(
        input_root,
        output_root,
        recursive,
        True,
        num_batches,
        batch_id,
        resume,
    )
    console.print(f"Completed {len(results)} videos")


@app.command("sample-frames")
def sample_frames(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    cfg = _config(config).with_overrides(
        **{
            "detection.enabled": False,
            "segmentation.enabled": False,
            "tracking.enabled": False,
            "ocr.enabled": False,
            "audio.enabled": False,
            "output.excel": True,
        }
    )
    result = LegoPoliticsAnalyzer(cfg).analyze_video(video, output)
    console.print(f"Sampled {len(result.frames)} frames")


def _run_feature(
    video: Path,
    output: Path,
    config: Path | None,
    overrides: dict,
) -> None:
    cfg = _config(config).with_overrides(**overrides)
    LegoPoliticsAnalyzer(cfg).analyze_video(video, output)


@app.command("detect")
def detect(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(video, output, config, {"detection.enabled": True})


@app.command("segment")
def segment(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(
        video,
        output,
        config,
        {
            "detection.enabled": True,
            "segmentation.enabled": True,
        },
    )


@app.command("track")
def track(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(
        video,
        output,
        config,
        {
            "detection.enabled": True,
            "tracking.enabled": True,
        },
    )


@app.command("describe")
def describe(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(
        video,
        output,
        config,
        {"vision.models.florence2.enabled": True},
    )


@app.command("ocr")
def ocr(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(video, output, config, {"ocr.enabled": True})


@app.command("transcribe")
def transcribe(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(video, output, config, {"audio.enabled": True})


@app.command("analyze-audio")
def analyze_audio(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(
        video,
        output,
        config,
        {
            "audio.enabled": True,
            "audio.audio_events.enabled": True,
        },
    )


@app.command("analyze-narrative")
def analyze_narrative(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    _run_feature(
        video,
        output,
        config,
        {"prompting.narrative_pass": True},
    )


@app.command("export")
def export(
    result_json: Annotated[
        Path,
        typer.Option(
            "--result-json",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    result = load_result(result_json)
    artifacts = export_result(result, _config(config))
    console.print({key: str(value) for key, value in artifacts.items()})


@app.command("validate")
def validate(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    required = [
        output / "result.json",
        output / "run_manifest.sqlite",
        output / "video_summary.xlsx",
    ]
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        console.print({"valid": False, "missing": missing})
        raise typer.Exit(1)

    load_result(output / "result.json")
    console.print({"valid": True, "output": str(output)})


@app.command("review")
def review(
    validation_file: Path,
    corrections_file: Path = Path("corrections.jsonl"),
) -> None:
    if importlib.util.find_spec("streamlit") is None:
        console.print("Install legopolitics[review]")
        raise typer.Exit(1)

    script = Path(__file__).parent / "review" / "streamlit_entry.py"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(script),
            "--",
            "--validation",
            str(validation_file),
            "--corrections",
            str(corrections_file),
        ],
        check=True,
    )


@app.command("inspect-run")
def inspect_run(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    with RunManifest(manifest) as database:
        console.print_json(data=database.inspect())


@app.command("retry-failures")
def retry_failures(
    video: Annotated[
        Path,
        typer.Option(
            "--video",
            "-v",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ],
    config: Path | None = None,
) -> None:
    cfg = _config(config).with_overrides(**{"project.resume": False})
    LegoPoliticsAnalyzer(cfg).analyze_video(video, output)


@app.command("migrate-output")
def migrate(
    path: Annotated[
        Path,
        typer.Option(
            "--path",
            exists=True,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    console.print({"schema_version": migrate_output(path)})


@app.command("list-models")
def list_models() -> None:
    table = Table("Capability", "Built-in adapter", "Optional extra")

    for row in [
        ("Detection", "YOLO, Grounding DINO, mock", "yolo / grounding"),
        (
            "Vision",
            "Florence-2, BLIP, BLIP-2, CLIP, DINOv2, generic VLM",
            "huggingface",
        ),
        ("Segmentation", "SAM 2, box-mask fallback", "segmentation"),
        (
            "Audio",
            "Whisper, Faster-Whisper, pyannote, audio classification",
            "audio / diarization",
        ),
        ("OCR", "Florence-2, TrOCR", "ocr"),
    ]:
        table.add_row(*row)

    console.print(table)


@app.command("inspect-yolo-model")
def inspect_yolo_model(
    weights: Annotated[
        Path,
        typer.Option(
            "--weights",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Inspect a custom YOLO checkpoint and print its task and class map."""
    try:
        console.print_json(data=inspect_yolo_weights(weights))
    except Exception as exc:
        console.print(f"[red]Could not inspect YOLO model:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command("register-yolo-model")
def register_yolo_model(
    weights: Annotated[
        Path,
        typer.Option(
            "--weights",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    destination: Annotated[
        Path,
        typer.Option(
            "--destination",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path("models/lego_figure_best.pt"),
    overwrite: bool = False,
) -> None:
    """Copy and register a custom spatial YOLO checkpoint."""
    try:
        console.print_json(
            data=register_yolo_weights(
                weights,
                destination,
                overwrite=overwrite,
            )
        )
    except Exception as exc:
        console.print(f"[red]Could not register YOLO model:[/red] {exc}")
        raise typer.Exit(1) from exc


@app.command("recommend-vlm-profile")
def recommend_vlm_profile() -> None:
    """Recommend a Hugging Face VLM profile for the current hardware."""
    info = detect_device("auto")
    profile = recommend_hardware_profile(info)

    profile_map = {
        "cpu": "hf-cpu",
        "small_gpu": "hf-small-gpu",
        "midrange_gpu": "hf-midrange-gpu",
        "high_end_gpu": "hf-high-end-gpu",
        "maximum": "hf-maximum",
    }

    console.print_json(
        data={
            "device": info.device,
            "gpu_name": info.gpu_name,
            "gpu_memory_gb": info.total_gpu_memory_gb,
            "bf16_supported": info.bf16_supported,
            "recommended_hardware_profile": profile,
            "config_template": profile_map[profile],
            "command": (
                f"legopolitics config-template "
                f"--profile {profile_map[profile]} "
                "--output analysis.yaml"
            ),
        }
    )


@app.command("doctor")
def doctor(output: Path = Path(".")) -> None:
    info = inspect_environment()
    info["output_writable"] = output.exists() and output.is_dir()

    review = Path("TRADEMARK_REVIEW.md")
    info["trademark_review_present"] = review.exists()
    info["trademark_review_completed"] = (
        "completed: true" in review.read_text(encoding="utf-8").lower()
        if review.exists()
        else False
    )

    console.print_json(data=info)


@app.command("benchmark")
def benchmark(
    output: Path = Path("benchmark_output"),
    seconds: int = 2,
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    video = output / "synthetic.mp4"
    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")  # noqa: B009
    writer = cv2.VideoWriter(str(video), fourcc, 10, (320, 240))

    for index in range(seconds * 10):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.rectangle(
            frame,
            (20 + index * 3, 60),
            (100 + index * 3, 180),
            (0, 255, 255),
            -1,
        )
        writer.write(frame)

    writer.release()

    start = time.perf_counter()

    cfg = AnalysisConfig().with_overrides(
        **{
            "sampling.mode": "fixed",
            "sampling.fps": 2.0,
            "detection.enabled": True,
            "detection.primary_backend": "mock",
            "output.parquet": False,
        }
    )

    result = LegoPoliticsAnalyzer(cfg).analyze_video(
        video,
        output / "results",
    )
    elapsed = time.perf_counter() - start

    console.print(
        {
            "elapsed_seconds": elapsed,
            "sampled_frames": len(result.frames),
            "detections": len(result.detections),
            "frames_per_second": (
                len(result.frames) / elapsed if elapsed else None
            ),
        }
    )


@app.command("config-template")
def config_template(
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = Path("analysis.yaml"),
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help=(
                "Configuration profile: default, lego-yolo, hf-ensemble, "
                "hf-cpu, hf-small-gpu, hf-midrange-gpu, hf-high-end-gpu, "
                "or hf-maximum."
            ),
        ),
    ] = "default",
) -> None:
    console.print(str(write_default_config(output, profile=profile)))


@app.command("schema")
def schema() -> None:
    console.print_json(data=AnalysisConfig.model_json_schema())


@app.command("about")
def about() -> None:
    console.print(f"legopolitics {__version__}\n\n{DISCLAIMER}")


@app.command("version")
def version() -> None:
    console.print(__version__)


if __name__ == "__main__":
    app()