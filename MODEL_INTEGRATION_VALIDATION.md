# Model Integration Validation

Date: 2026-07-12

## Attached archive inspection

`yolo26s_results.zip` was inspected directly.

- `yolo26s_results/weights/best.pt`: 0 bytes
- `yolo26s_results/weights/last.pt`: 0 bytes
- `yolo26s_results/args.yaml`: 0 bytes
- `yolo26s_results/results.csv`: 0 bytes
- other training images and charts: 0 bytes
- nonzero content: macOS `__MACOSX/._*` metadata entries only

Conclusion: the uploaded archive does not contain a usable checkpoint. The real
nonzero `best.pt` must be re-uploaded before model task, class map, or inference
can be verified.

## Implemented YOLO integration

- zero-byte and missing-file rejection;
- checkpoint task validation;
- classification-checkpoint rejection for spatial extraction;
- task and class-map inspection command;
- atomic model registration command;
- SHA-256 checkpoint fingerprinting;
- model metadata JSON sidecar;
- external `models/` workflow rather than wheel embedding;
- detector configuration and tutorial.

Commands:

```bash
legopolitics inspect-yolo-model --weights /path/to/best.pt
legopolitics register-yolo-model --weights /path/to/best.pt \
  --destination models/lego_figure_best.pt
```

## Implemented Hugging Face VLM integration

- generic `image-text-to-text` adapter;
- multiple VLMs in one YAML configuration;
- SmolVLM 500M;
- SmolVLM2 500M and 2.2B;
- Qwen2.5-VL 3B;
- Idefics3 8B;
- LLaVA-OneVision 7B;
- CPU, small-GPU, midrange, high-end, and maximum profiles;
- FP32, FP16, and BF16;
- bitsandbytes 4-bit and 8-bit loading;
- Accelerate device maps;
- CPU and disk offload;
- explicit per-device memory limits;
- optional FlashAttention 2;
- sequential model unloading;
- hardware-profile recommendation command;
- provenance recording for dtype, quantization, device map, and profile.

## Validation commands and outcomes

```text
PYTHONPATH=src pytest -q
40 passed

ruff format --check src tests
188 files already formatted

ruff check src tests
All checks passed

Targeted mypy check of modified modules
Success: no issues found in 6 source files

mkdocs build --strict
Passed

python -m build
Built wheel and source distribution

python -m twine check --strict dist/*
Both distributions passed

Fresh-wheel command validation
Version, packaged GPU profile, and register-yolo-model help passed
```

## Not executed

Real YOLO inference was not executed because the supplied checkpoint is empty.
Real VLM inference was not executed because model downloads, CUDA hardware, and
large external weights were not available in this build environment. The
adapters and loading plans are covered through configuration and unit tests.
