<p align="center">
  <img src="https://raw.githubusercontent.com/cantayus/legopolitics/main/docs/assets/legopolitics-logo.png" alt="legopolitics logo" width="360">
</p>

<h1 align="center">legopolitics</h1>

<p align="center">
  A multimodal Python research package for analyzing political narratives in brick-built videos.
</p>

<p align="center">
  <a href="https://github.com/cantayus/legopolitics/actions/workflows/ci.yml"><img src="https://github.com/cantayus/legopolitics/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://cantayus.github.io/legopolitics/"><img src="https://github.com/cantayus/legopolitics/actions/workflows/pages.yml/badge.svg" alt="Documentation"></a>
  <a href="https://pypi.org/project/legopolitics/"><img src="https://img.shields.io/pypi/v/legopolitics.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/legopolitics/"><img src="https://img.shields.io/pypi/pyversions/legopolitics.svg" alt="Python versions"></a>
</p>

`legopolitics` converts videos made with brick-built figures, miniature environments, and stop-motion scenes into structured, reproducible research data. It combines video sampling, object detection, tracking, image-language models, OCR, audio analysis, narrative coding, political measurement, model agreement, and human validation in one extensible pipeline.

> **Trademark notice:** LEGO® is a trademark of the LEGO Group of companies, which does not sponsor, authorize, or endorse this project. legopolitics is an independent academic research software project and is not affiliated with, maintained by, sponsored by, authorized by, or associated with the LEGO Group.

## Why use legopolitics?

Political brick-built videos can communicate through visual composition, recurring characters, symbols, dialogue, narration, music, violence, camera framing, and sequential storytelling. A single caption per frame cannot preserve all of those signals. `legopolitics` therefore separates the pipeline into linked units of analysis:

- **Videos and scenes** — metadata, scene boundaries, sampling decisions, and narrative segments.
- **Frames** — visual descriptions, composition, salience, OCR, transcript alignment, and research coding.
- **Figures and objects** — detections, crops, masks, attributes, tracks, group assignments, and roles.
- **Interactions** — figure–figure and figure–object relationships over time.
- **Audio** — transcription, translation, diarization, music, silence, and sound events.
- **Interpretation** — blind observation, context-conditioned analysis, codebook scoring, contradiction checks, and model agreement.

## Capabilities

| Area | Included capabilities |
|---|---|
| Video processing | Recursive discovery, FFmpeg/OpenCV probing, fixed/scene/hybrid/adaptive/all-frame sampling, perceptual deduplication, motion and quality metrics |
| Detection | Custom YOLO-compatible weights, open-vocabulary detection, detection fusion, annotations, crops, and configurable taxonomies |
| Temporal analysis | Figure/object tracking, track reactivation, appearance-assisted re-identification, contact sheets, temporal events, and narrative segments |
| Segmentation | Detector masks, box-mask fallback, and optional SAM-style segmentation/propagation adapters |
| Visual models | Florence-2, BLIP, BLIP-2, CLIP, DINOv2, generic Hugging Face VLMs, video VLMs, and remote-provider adapters |
| Text and audio | OCR, subtitle deduplication, Whisper/Faster-Whisper, translation, speaker diarization, and audio-event adapters |
| Political analysis | Roles, agency, threat, heroism, victimization, retaliation, legitimacy, affect, dehumanization, and researcher-defined codebooks |
| Reliability | Majority/max/weighted voting, model disagreement, uncertainty, validation sampling, manual corrections, and agreement metrics |
| Outputs | Excel, Parquet, JSONL, SQLite, HTML reports, frames, crops, masks, prompts, logs, configuration, and provenance |

Heavy machine-learning dependencies are optional and loaded lazily. The base package can run video discovery, probing, sampling, quality analysis, deterministic measurements, manifests, and exports without downloading large models.

## Installation

After the first PyPI release:

```bash
python -m pip install legopolitics
```

Install only the optional capabilities required for your project:

```bash
python -m pip install "legopolitics[yolo,huggingface,parquet]"
python -m pip install "legopolitics[audio,diarization,ocr]"
python -m pip install "legopolitics[all]"
```

For development from GitHub:

```bash
git clone https://github.com/cantayus/legopolitics.git
cd legopolitics
python -m pip install -e ".[dev]"
```

FFmpeg is recommended for reliable metadata probing and audio extraction.

Ultralytics is intentionally optional. Review its current licensing terms before enabling the YOLO adapter.

## Quick start

### 1. Check the environment

```bash
legopolitics doctor
```

### 2. Create a configuration

```bash
legopolitics config-template --output analysis.yaml
```

A safe, lightweight example is also available at [`configs/minimal.yaml`](configs/minimal.yaml). A full research configuration is available at [`configs/comprehensive.yaml`](configs/comprehensive.yaml).

### 3. Analyze a video

```bash
legopolitics analyze-video \
  --video "videos/example.mp4" \
  --config "configs/minimal.yaml" \
  --output "outputs/example"
```

### 4. Use the Python API

```python
from pathlib import Path

from legopolitics import AnalysisConfig, LegoPoliticsAnalyzer

config = AnalysisConfig.from_yaml(Path("configs/minimal.yaml"))
analyzer = LegoPoliticsAnalyzer(config)

result = analyzer.analyze_video(
    video_path=Path("videos/example.mp4"),
    output_dir=Path("outputs/example"),
)

print(f"Sampled frames: {len(result.frames)}")
print(f"Detections: {len(result.detections)}")
print(f"Tracks: {len(result.tracks)}")
print(result.output_paths)
```

## Custom YOLO model

The currently supplied ZIP contains zero-byte placeholders rather than the real checkpoint. Re-create the archive and confirm that `weights/best.pt` has a nonzero size. Then register the real model:

```bash
legopolitics register-yolo-model \
  --weights /path/to/yolo26s_results/weights/best.pt \
  --destination models/lego_figure_best.pt
```

The command validates the checkpoint, prints its task and class map, fingerprints it, copies it atomically, and writes a JSON metadata sidecar. You can inspect without copying:

```bash
legopolitics inspect-yolo-model --weights /path/to/best.pt
```

A checkpoint that reports `classify` cannot extract figures from complete frames; use the trained `detect` checkpoint instead. Start from [`configs/lego_figure_detector.yaml`](configs/lego_figure_detector.yaml):

```yaml
detection:
  enabled: true
  primary_backend: "yolo"
  yolo:
    weights: "models/lego_figure_best.pt"
    task: "detect"
    confidence_threshold: 0.30
    iou_threshold: 0.50
    image_size: 1280

legal:
  ultralytics_license_acknowledged: true
```

Then run:

```bash
legopolitics analyze-video \
  --video "videos/example.mp4" \
  --config "configs/lego_figure_detector.yaml" \
  --output "outputs/example"
```

See the [custom YOLO integration guide](docs/custom_yolo_model.md) for validation and threshold-tuning guidance.

## Hugging Face VLM ensemble

The package can run several Hugging Face visual-language models sequentially and retain every output separately. The included ensemble configuration integrates:

- `HuggingFaceTB/SmolVLM2-2.2B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct`
- `HuggingFaceM4/Idefics3-8B-Llama3`
- `llava-hf/llava-onevision-qwen2-7b-ov-hf`

Use [`configs/huggingface_vlm_ensemble.yaml`](configs/huggingface_vlm_ensemble.yaml), or combine it with the detector settings in [`configs/lego_figure_detector.yaml`](configs/lego_figure_detector.yaml). SmolVLM2 and Qwen2.5-VL are enabled in that workstation-oriented example; the larger models are integrated but disabled by default.

The package supports CPU, small-GPU, midrange, high-end, and maximum-quality profiles. Ask the CLI for a recommendation:

```bash
legopolitics recommend-vlm-profile
legopolitics config-template --profile hf-midrange-gpu --output analysis.yaml
```

Available templates include `hf-cpu`, `hf-small-gpu`, `hf-midrange-gpu`, `hf-high-end-gpu`, and `hf-maximum`. They use smaller checkpoints, FP16/BF16, 4-bit loading, Accelerate device maps, and sequential unloading as appropriate. See the [Hugging Face VLM ensemble guide](docs/huggingface_vlm_ensemble.md).

## Research context and interpretation

The package keeps visual observation separate from political interpretation. Add carefully sourced context and a research question to the configuration:

```yaml
research:
  event_context: >
    Add the factual historical context needed to interpret the video.
  research_question: >
    How are the depicted sides presented in terms of agency,
    aggression, victimization, heroism, competence, and legitimacy?
  codebook_path: "configs/propaganda_codebook.yaml"

prompting:
  blind_visual_pass: true
  contextual_pass: true
  codebook_pass: true
  contradiction_pass: true
```

The blind pass receives no event context. The contextual pass receives the supplied background but must distinguish direct evidence from inference. The contradiction pass looks for evidence that weakens the proposed interpretation.

## Outputs

A typical output directory contains:

```text
outputs/example/
├── video_summary.xlsx
├── report.html
├── result.json
├── run_manifest.sqlite
├── effective_config.yaml
├── environment.json
├── provenance.json
├── tables/
├── jsonl/
├── frames/
├── annotated_frames/
├── crops/
├── masks/
├── validation/
└── logs/
```

The Excel workbook provides linked sheets for frames, detections, tracks, relationships, OCR, transcripts, audio events, narrative segments, codebook scores, model agreement, validation, metadata, prompts, models, and errors. Parquet and SQLite are the preferred machine-readable formats for larger studies.

## Batch processing

```bash
legopolitics analyze-directory \
  --input-root "videos" \
  --output-root "outputs" \
  --recursive \
  --num-batches 4 \
  --batch-id 0 \
  --resume
```

Run the same command in four terminals or notebooks with batch IDs `0`, `1`, `2`, and `3`. Video assignment is deterministic, and the manifest prevents completed stages from being unnecessarily recomputed.

## Documentation and tutorial

- [Documentation site](https://cantayus.github.io/legopolitics/)
- [Complete tutorial](docs/tutorial.md)
- [Python API](docs/python_api.md)
- [CLI reference](docs/cli.md)
- [Configuration reference](docs/configuration.md)
- [Publication guide](PUBLISHING.md)
- [Methodology and safeguards](docs/methodology.md)

## Development

```bash
python -m pip install -e ".[dev]"
ruff format --check src tests
ruff check src tests
mypy src/legopolitics
python -m pytest
mkdocs build --strict
python -m build
python -m twine check --strict dist/*
```

## Project status

`legopolitics` is currently an alpha research package. Optional adapters for large models require compatible hardware, model weights, and—in some cases—gated repository access or provider credentials. Model outputs should be validated against human coding before they are treated as substantive research measurements.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Please cite the exact package version, model revisions, custom weights, prompts, and configuration used in an analysis.

## License and public release

The original `legopolitics` source code and project-authored documentation are distributed under the [MIT License](LICENSE). The MIT License permits use, modification, publication, distribution, sublicensing, and sale while requiring preservation of the copyright and license notices.

Third-party packages, external model repositories, datasets, and model weights remain subject to their own licenses. In particular, the optional Ultralytics adapter and the separately distributed YOLO checkpoint require an independent license review; the package MIT License does not override those terms. See [`NOTICE`](NOTICE), [`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md), [`MODEL_LICENSES.md`](MODEL_LICENSES.md), and [`TRADEMARK_REVIEW.md`](TRADEMARK_REVIEW.md).

## Included research detector

The source-repository bundle includes the supplied YOLO26s spatial detector under `model_assets/lego_figure_yolo26s/`. It detects `human` and `lego_figure`; the ready configuration `configs/lego_figure_yolo26s.yaml` selects only `lego_figure` for extraction, tracking, and VLM analysis. The checkpoint is intentionally excluded from the PyPI wheel and should be managed with Git LFS when the repository is published.
