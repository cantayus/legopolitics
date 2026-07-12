# Implementation Status

## Fully exercised core

The following capabilities were executed through automated tests or an installed-wheel integration run:

- Recursive discovery and deterministic multi-batch assignment
- FFprobe/OpenCV video probing
- Fixed, scene, hybrid, adaptive, and all-frame sampling
- Frame hashing, motion, scene, and quality metrics
- Mock detection, box fusion, crops, box-mask segmentation, IoU tracking, salience, relations, and track events
- Versioned Pydantic configuration and schemas
- Stage cache, resume behavior, SQLite manifest, locks, errors, and artifacts
- Excel, Parquet, JSONL, SQLite, HTML, configuration, environment, and provenance output
- Model-agreement utilities, validation sampling, codebook loading, OCR deduplication, structured-output parsing, and security sanitization
- CLI, doctor, benchmark, package build, documentation build, and release gate

## Implemented optional adapters not executed with large weights

The repository includes lazy, typed adapters and dependency checks for:

- User-supplied Ultralytics-compatible YOLO weights and native YOLO tracking
- Grounding DINO / Transformers zero-shot detection
- SAM 2 segmentation
- Florence-2, BLIP, BLIP-2, CLIP, DINOv2, generic Hugging Face VLMs, video VLMs, and OpenAI-compatible endpoints
- Florence-2 OCR, TrOCR, and callable provider OCR
- Whisper, Faster-Whisper, pyannote diarization, translation, and audio-event classification
- Streamlit human-review interface

These heavy integrations were not run because the build environment did not provide model weights, CUDA hardware, gated tokens, or remote API credentials. Their imports are lazy, failures are explicit, and standard CI uses mocks rather than downloading multi-gigabyte models.

## Deliberate limitations

- SAM 3 is exposed as a plugin gate rather than guessing an unstable API. It raises an actionable dependency error and points to the tracker/segmenter entry-point groups.
- Portable `ByteTrackAdapter` and `BoTSORTAdapter` operate on table-level detections through the tested association core; native algorithms are available through `YoloTrackAdapter` when Ultralytics is installed.
- The original package source code and project-authored documentation are licensed under the MIT License. Public release remains blocked until the trademark review is completed and third-party dependency/model obligations are confirmed.
- Automated political interpretation is not treated as ground truth. The package preserves blind/contextual separation, evidence, contradictions, alternatives, provenance, and manual corrections.
