# Quick Start

## Install

After the package is released on PyPI:

```bash
python -m pip install legopolitics
```

For a local source checkout:

```bash
python -m pip install -e ".[dev]"
```

## Check the environment

```bash
legopolitics doctor
```

This reports Python, FFmpeg, PyTorch, CUDA, optional adapters, output-directory access, and release-review status without exposing secret tokens.

## Run the lightweight pipeline

```bash
legopolitics analyze-video \
  --video input.mp4 \
  --config configs/minimal.yaml \
  --output outputs/input
```

The minimal configuration avoids large model downloads. It performs probing, hybrid frame sampling, quality and composition measurements, manifests, and research exports.

## Inspect the result

```bash
legopolitics validate --output outputs/input
legopolitics inspect-run --manifest outputs/input/run_manifest.sqlite
```

Open `outputs/input/video_summary.xlsx` for the researcher-facing workbook and `outputs/input/report.html` for the local report.

## Python API

```python
from pathlib import Path

from legopolitics import AnalysisConfig, LegoPoliticsAnalyzer

config = AnalysisConfig.from_yaml(Path("configs/minimal.yaml"))
result = LegoPoliticsAnalyzer(config).analyze_video(
    video_path=Path("input.mp4"),
    output_dir=Path("outputs/input"),
)

print(result.output_paths)
```

Continue with the [complete tutorial](tutorial.md) to add a custom detector, tracking, vision-language models, OCR, audio, research context, narrative coding, and validation.
