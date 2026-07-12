# Installation

Base install:

```bash
python -m pip install .
```

Editable development install:

```bash
python -m pip install -e ".[dev]"
```

Optional components are installed separately, for example `.[yolo]`, `.[huggingface]`, `.[audio]`, `.[diarization]`, `.[parquet]`, or `.[all]`. Install FFmpeg separately for audio extraction and the most reliable metadata probing.

Ultralytics and every model repository have independent licensing terms. Review them before use.
