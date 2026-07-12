# Custom YOLO and Hugging Face VLM Integration Guide

## Finding from the attached YOLO archive

The inspected `yolo26s_results.zip` contains the normal Ultralytics result
filenames, but every actual result file is zero bytes, including:

```text
yolo26s_results/weights/best.pt
yolo26s_results/weights/last.pt
yolo26s_results/args.yaml
yolo26s_results/results.csv
```

The nonzero entries are macOS `__MACOSX/._*` metadata files, not model weights.
The current archive therefore cannot be integrated or inspected as a model.

## Re-create and verify the archive

Confirm the real checkpoint first:

```bash
ls -lh /path/to/yolo26s_results/weights/best.pt
```

The size must be greater than zero. Then create a fresh archive without macOS
metadata:

```bash
cd /path/to/parent
zip -r yolo26s_results_complete.zip yolo26s_results \
  -x "*/__MACOSX/*" "*/.DS_Store"
unzip -l yolo26s_results_complete.zip | grep best.pt
```

## Install and register the model

```bash
python -m pip install -e ".[yolo]"

legopolitics register-yolo-model \
  --weights /path/to/yolo26s_results/weights/best.pt \
  --destination models/lego_figure_best.pt
```

The command creates:

```text
models/lego_figure_best.pt
models/lego_figure_best.pt.json
```

The JSON sidecar records the checkpoint hash, task, and class map.

## Configure the detector

```yaml
detection:
  enabled: true
  primary_backend: yolo
  yolo:
    weights: models/lego_figure_best.pt
    task: auto
    confidence_threshold: 0.30
    iou_threshold: 0.50
    image_size: 1280
    device: auto
    half_precision: true
```

Run a short validation video before the complete corpus:

```bash
legopolitics detect \
  --video videos/test.mp4 \
  --config configs/lego_figure_detector.yaml \
  --output outputs/yolo_test
```

## Select a VLM profile for the machine

```bash
legopolitics recommend-vlm-profile
```

Generate the recommended template, for example:

```bash
legopolitics config-template \
  --profile hf-midrange-gpu \
  --output configs/my_vlm_config.yaml
```

Available profiles:

```text
hf-cpu
hf-small-gpu
hf-midrange-gpu
hf-high-end-gpu
hf-maximum
```

## Integrated Hugging Face VLMs

The package includes generic adapters and examples for:

```text
HuggingFaceTB/SmolVLM-500M-Instruct
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
HuggingFaceTB/SmolVLM2-2.2B-Instruct
Qwen/Qwen2.5-VL-3B-Instruct
HuggingFaceM4/Idefics3-8B-Llama3
llava-hf/llava-onevision-qwen2-7b-ov-hf
```

Install standard support:

```bash
python -m pip install -e ".[huggingface]"
```

Install quantized loading support for lower-memory GPUs:

```bash
python -m pip install -e ".[huggingface,quantization]"
```

The package supports FP32, FP16, BF16, 4-bit, 8-bit, Accelerate device maps,
CPU/disk offload, explicit memory limits, and sequential VLM unloading.

## Combined configuration

Start with the YOLO configuration, then copy the `vision` and `performance`
blocks from the appropriate Hugging Face profile into the same YAML file.
The package will:

1. sample the video;
2. detect figures with the custom YOLO checkpoint;
3. save bounding boxes and crops;
4. run selected VLMs on whole frames and figure crops;
5. preserve each model's raw response;
6. calculate agreement and voting outputs;
7. export Excel, Parquet, JSONL, SQLite, and HTML results.
